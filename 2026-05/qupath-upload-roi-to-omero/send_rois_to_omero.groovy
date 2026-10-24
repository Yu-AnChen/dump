/**
 * send_rois_to_omero.groovy
 *
 * Sends QuPath ROIs to OMERO via ome-omero-roitool. Reverse: pull_rois_from_omero.groovy.
 *
 * HOLE_MODE picks how polygon holes (interior rings) are written (ADR-017, ADR-023):
 *   "exclude" — a separate ROI named "Exclude"; Pageant subtracts spatially contained
 *               Exclude ROIs from their enclosing ROI when computing measurements
 *   "union"   — ".hole." polygon shapes inside the object's own ROI
 * pull_rois_from_omero.groovy reads both back into holes.
 *
 * What this script does:
 *   1. Downloads and caches ome-omero-roitool (once, ~77MB)
 *   2. Downloads and caches a JRE from Adoptium (once, ~50MB) — no manual Java install needed
 *   3. Extracts OMERO server host and image ID from the currently open image URI
 *   4. Resolves credentials: cached ICE session key → password prompt → web login for session key
 *   5. Exports all annotations + detections to a temp OME-XML file
 *      (polygon holes written per HOLE_MODE)
 *   6. Runs ome-omero-roitool as a subprocess (keeps QuPath alive)
 *   7. Cleans up the temp file
 *
 * Requirements:
 *   - QuPath 0.6+ with the OMERO extension installed
 *   - Current image must be open from an OMERO server
 *   - OMERO ICE port 4064 must be reachable
 *   - Internet access on first run (to download tool + JRE)
 */

// ── User-configurable ──────────────────────────────────────────────────────────
// ==== SHARED BEGIN config — keep identical in push/pull scripts (check_shared.py) ====
ROITOOL_VERSION  = "0.2.7"
OMERO_PORT       = 4064
JRE_VERSION      = "21"
// Set if your OMERO Blitz/ICE server has a different hostname than the web
// frontend. Leave as "" to use the cached host, else the image URL's host; a wrong
// host prompts once and the answer is cached (ADR-015).
OMERO_ICE_HOST   = ""
// Heap for the roitool subprocess. Images with many thousands of ROIs need more
// than the JVM default; raise if roitool dies with OutOfMemoryError.
SUBPROCESS_HEAP  = "4g"
// Total wall-clock budget for one roitool run. roitool prints NOTHING while it
// transfers, so this cannot be an idle timeout. Measured: 1118 ROIs took 244s to
// export from a LAN OMERO server, ~0.2s per ROI (ADR-021). Raise for huge images.
ROITOOL_TIMEOUT_SEC = 1800
// ==== SHARED END config ====
// How polygon holes are written: "exclude" (Pageant) or "union" — see the header.
HOLE_MODE = "exclude"
// ──────────────────────────────────────────────────────────────────────────────

import java.net.URI
import java.net.URL
import java.util.concurrent.TimeUnit
import java.util.zip.*
import javafx.application.Platform
import javafx.geometry.Insets
import javafx.scene.control.CheckBox
import javafx.scene.control.Label
import javafx.scene.control.PasswordField
import javafx.scene.control.TextField
import javafx.scene.layout.VBox
import javafx.stage.FileChooser
import ome.specification.XMLWriter
import ome.units.UNITS
import ome.units.quantity.Length
import ome.xml.model.*
import ome.xml.model.enums.FillRule
import ome.xml.model.primitives.Color
import ome.xml.model.primitives.NonNegativeInteger
import qupath.lib.common.ColorTools
import qupath.lib.gui.prefs.PathPrefs
import qupath.lib.scripting.QP
import qupath.lib.io.PathIO
import qupath.lib.objects.PathROIObject
import qupath.lib.roi.*

boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win")
def userDir = PathPrefs.userPathProperty().get() ?:
    new File(System.getProperty("user.home"), ".qupath-roitool").absolutePath
new File(userDir).mkdirs()
if (!(HOLE_MODE in ["exclude", "union"]))
    throw new IllegalArgumentException("HOLE_MODE must be \"exclude\" or \"union\", not \"${HOLE_MODE}\"")

// ── Self-test: hole writing in both HOLE_MODEs, no OMERO or image needed ───────
// QuPath CLI: QuPath script send_rois_to_omero.groovy -a --selftest
if (binding.hasVariable('args') && (args as List).contains('--selftest')) {
    def gf = GeometryTools.getDefaultFactory()
    def ring = { double a, double b -> gf.createLinearRing([[a, a], [b, a], [b, b], [a, b], [a, a]]
        .collect { new org.locationtech.jts.geom.Coordinate(it[0], it[1]) } as org.locationtech.jts.geom.Coordinate[]) }
    def holed = gf.createPolygon(ring(0, 100), [ring(25, 75)] as org.locationtech.jts.geom.LinearRing[])
    def obj = qupath.lib.objects.PathObjects.createAnnotationObject(
        GeometryTools.geometryToROI(holed, qupath.lib.regions.ImagePlane.getDefaultPlane()),
        qupath.lib.objects.classes.PathClass.fromString("Stroma"))
    // mode → [ROIs emitted, ROI names, ".hole." shapes per ROI]
    [exclude: [2, ["Stroma", "Exclude"], [0, 1]],
     union:   [1, ["Stroma"],            [1]]].each { mode, want ->
        HOLE_MODE = mode
        def ome = new OME()
        int n = addROI(obj.getROI(), obj, 0, ome, new StructuredAnnotations())
        def rois = ome.copyROIList()
        def holes = rois.collect { r -> r.union.copyShapeList().count { it.ID.contains(".hole.") } }
        assert [n, rois*.name, holes] == want : "HOLE_MODE ${mode}: got ${[n, rois*.name, holes]}, want ${want}"
    }
    println "selftest OK"
    return
}

// ==== SHARED BEGIN session-cache — keep identical in push/pull scripts (check_shared.py) ====
// Session cache shared by the push and pull scripts (ADR-012): ICE host, username
// and session key — never the password.
def credsFile   = new File(userDir, ".omero_qupath_creds")
def cachedProps = new Properties()
if (credsFile.exists()) credsFile.withInputStream { cachedProps.load(it) }

def saveCache = { String iceHost, String user, String key ->
    def props = new Properties()
    props.setProperty("ice_host",    iceHost ?: "")
    props.setProperty("username",    user ?: "")
    props.setProperty("session_key", key ?: "")
    credsFile.withOutputStream { props.store(it, "QuPath OMERO session cache — delete to reset") }
}
// ==== SHARED END session-cache ====

// ==== SHARED BEGIN provisioning — keep identical in push/pull scripts (check_shared.py) ====
// ── Download helper ───────────────────────────────────────────────────────────
def downloadWithProgress = { url, dest, label ->
    print("Downloading ${label}...")
    def conn = new URL(url.toString()).openConnection()
    conn.connect()
    long total      = conn.contentLengthLong
    long downloaded = 0
    long nextReport = 5 * 1024 * 1024
    def buf = new byte[65536]
    conn.inputStream.withCloseable { is ->
        dest.withOutputStream { os ->
            int n
            while ((n = is.read(buf)) != -1) {
                os.write(buf, 0, n)
                downloaded += n
                if (downloaded >= nextReport) {
                    def mb      = (long)(downloaded / (1024 * 1024))
                    def totalMb = total > 0 ? " / ${(long)(total / (1024 * 1024))} MB" : " MB"
                    print("  ${mb}${totalMb}")
                    nextReport += 5 * 1024 * 1024
                }
            }
        }
    }
    print("  done (${String.format('%.1f', downloaded / 1024.0 / 1024.0)} MB)")
}

// ── Resolve roitool binary (ADR-008) ──────────────────────────────────────────
def toolDir = new File(userDir, "ome-omero-roitool-${ROITOOL_VERSION}")
def toolZip = new File(userDir, "ome-omero-roitool-${ROITOOL_VERSION}.zip")
// runRoitool starts roitool as plain Java on these jars; nothing else in the zip is used
def toolLib = new File(toolDir, "lib")

if (!toolLib.isDirectory()) {
    def downloadUrl = "https://github.com/glencoesoftware/ome-omero-roitool/releases/download/" +
                      "v${ROITOOL_VERSION}/ome-omero-roitool-${ROITOOL_VERSION}.zip"
    try {
        downloadWithProgress(downloadUrl, toolZip, "ome-omero-roitool v${ROITOOL_VERSION}")
        new ZipInputStream(new FileInputStream(toolZip)).withCloseable { zis ->
            def entry
            while ((entry = zis.nextEntry) != null) {
                def outFile = new File(userDir, entry.name)
                if (entry.directory) { outFile.mkdirs() }
                else { outFile.parentFile?.mkdirs(); outFile.withOutputStream { os -> os << zis } }
                zis.closeEntry()
            }
        }
        toolZip.delete()
        if (!toolLib.isDirectory()) {
            Dialogs.showErrorMessage("Install failed", "No ${toolLib} after extracting.")
            return
        }
        print("Downloaded roitool: ${toolDir.absolutePath}")
    } catch (Exception e) {
        Dialogs.showErrorMessage("Download failed", "Could not download ome-omero-roitool:\n${e.message}")
        return
    }
} else {
    print("Using cached roitool: ${toolDir.absolutePath}")
}

// ── Resolve Java: auto-downloaded Adoptium JRE (ADR-010) ──────────────────────
def osName    = System.getProperty("os.name").toLowerCase()
def osArch    = System.getProperty("os.arch").toLowerCase()
def adoptOs   = osName.contains("win") ? "windows" : osName.contains("mac") ? "mac" : "linux"
def adoptArch = (osArch.contains("aarch64") || osArch.contains("arm64")) ? "aarch64" : "x64"
def jreDirName  = "jre-${JRE_VERSION}-${adoptOs}-${adoptArch}"
def jreCacheDir = new File(userDir, jreDirName)
def jreExe    = new File(jreCacheDir, isWindows ? "bin/java.exe" : "bin/java")
def jreExeMac = new File(jreCacheDir, "Contents/Home/bin/java")
def javaHome  = null

if (jreExeMac.exists()) {
    javaHome = new File(jreCacheDir, "Contents/Home").absolutePath
    print("Using cached JRE (macOS): ${javaHome}")
} else if (jreExe.exists()) {
    javaHome = jreCacheDir.absolutePath
    print("Using cached JRE: ${javaHome}")
} else {
    def ext         = isWindows ? "zip" : "tar.gz"
    def downloadUrl = "https://api.adoptium.net/v3/binary/latest/${JRE_VERSION}/ga/${adoptOs}/${adoptArch}/jre/hotspot/normal/eclipse?project=jdk"
    def archiveFile = new File(userDir, "${jreDirName}.${ext}")
    try {
        downloadWithProgress(downloadUrl, archiveFile, "JRE ${JRE_VERSION} (${adoptOs}-${adoptArch})")
        print("Extracting JRE...")
        jreCacheDir.mkdirs()
        if (isWindows) {
            new ZipInputStream(new FileInputStream(archiveFile)).withCloseable { zis ->
                def entry
                while ((entry = zis.nextEntry) != null) {
                    def parts = entry.name.split("/", 2)
                    if (parts.length < 2 || !parts[1]) { zis.closeEntry(); continue }
                    def outFile = new File(jreCacheDir, parts[1])
                    if (entry.directory) { outFile.mkdirs() }
                    else { outFile.parentFile?.mkdirs(); outFile.withOutputStream { os -> os << zis } }
                    zis.closeEntry()
                }
            }
        } else {
            def GzipCI = Class.forName("org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream")
            def TarAIS  = Class.forName("org.apache.commons.compress.archivers.tar.TarArchiveInputStream")
            def gzIn  = GzipCI.getDeclaredConstructor(InputStream).newInstance(new FileInputStream(archiveFile))
            def tarIn = TarAIS.getDeclaredConstructor(InputStream).newInstance(gzIn)
            tarIn.withCloseable {
                def entry
                while ((entry = tarIn.nextEntry) != null) {
                    if (!tarIn.canReadEntryData(entry)) continue
                    def parts = entry.name.split("/", 2)
                    if (parts.length < 2 || !parts[1]) continue
                    def outFile = new File(jreCacheDir, parts[1])
                    if (entry.directory) { outFile.mkdirs() }
                    else {
                        outFile.parentFile?.mkdirs()
                        outFile.withOutputStream { os ->
                            def buf = new byte[8192]; int n
                            while ((n = tarIn.read(buf)) != -1) os.write(buf, 0, n)
                        }
                        if (entry.mode & 0111) outFile.setExecutable(true, false)
                    }
                }
            }
        }
        archiveFile.delete()
        if (jreExeMac.exists()) {
            javaHome = new File(jreCacheDir, "Contents/Home").absolutePath
        } else if (jreExe.exists()) {
            javaHome = jreCacheDir.absolutePath
        } else {
            Dialogs.showErrorMessage("JRE install failed",
                "Could not find java binary after extracting.\nCheck the QuPath log for details.")
            return
        }
        print("JRE ready: ${javaHome}")
    } catch (Exception e) {
        Dialogs.showErrorMessage("JRE download failed",
            "Could not download JRE from Adoptium:\n${e.message}")
        return
    }
}
// ==== SHARED END provisioning ====

// ==== SHARED BEGIN roitool-runner — keep identical in push/pull scripts (check_shared.py) ====
// ── roitool subprocess (ADR-009/016/021/026) ─────────────────────────────────
// roitool runs as a subprocess so ICE's System.exit() kills only the child, not QuPath.

// The password never goes on the command line, where any process can read it.
// roitool's --password is a picocli interactive option: given with no value it reads
// the password from stdin (runRoitool's stdinText). It must come last, because its
// optional value (arity 0..1) would otherwise swallow the image ID.
def roitoolAuthArgs = { String key, String user, String imageId, String path ->
    key ? ["--key", key, imageId, path] : ["--username", user, imageId, path, "--password"]
}

// subcommand: "import" or "export". stdinText: the password when not using a key, else null.
// roitool runs as plain Java with the lib/* classpath wildcard on every OS: no launcher
// script, no cmd.exe line-length limit, no environment to set up (ADR-026).
def runRoitool = { String subcommand, String iceHost, List authArgs, String stdinText ->
    def java = new File(javaHome.toString(), isWindows ? "bin/java.exe" : "bin/java").absolutePath
    def cmd = [java, "-Xmx" + SUBPROCESS_HEAP, "-cp", new File(toolLib, "*").absolutePath,
               "com.glencoesoftware.roitool.Main",
               subcommand, "--server", iceHost, "--port", OMERO_PORT.toString()] + authArgs
    def proc = new ProcessBuilder(cmd*.toString()).redirectErrorStream(true).start()
    proc.outputStream.withWriter("UTF-8") { w -> if (stdinText != null) w.write(stdinText + "\n") }
    def output = new StringBuilder()
    // ICE threads keep the JVM alive after roitool's main thread dies, so the
    // process never exits on a fatal error; kill it 5s after one is seen (ADR-016).
    // Any uncaught main-thread exception counts, e.g. "Obtained null object proxy"
    // from a wrong ICE host, not just the named connection/auth errors.
    def fatalSeen = new java.util.concurrent.atomic.AtomicBoolean(false)
    def reader = Thread.start {
        proc.inputStream.eachLine { line ->
            print(line); output.append(line).append("\n")
            if (line.startsWith("Exception in thread \"main\"") ||
                    line.contains("ConnectionRefusedException") ||
                    line.contains("PermissionDeniedException") ||
                    line.contains("Connection refused"))
                fatalSeen.set(true)
        }
    }
    long startMs = System.currentTimeMillis()
    long nextTick = 30000
    boolean timedOut = false
    while (proc.isAlive()) {
        if (proc.waitFor(500, TimeUnit.MILLISECONDS)) break
        long elapsedMs = System.currentTimeMillis() - startMs
        // Heartbeat: roitool is silent while transferring, so without this a slow
        // run is indistinguishable from a hang (ADR-021).
        if (elapsedMs >= nextTick) {
            print("  ...roitool ${subcommand} still running (${(long)(elapsedMs / 1000)}s elapsed)")
            nextTick += 30000
        }
        long elapsedSec = elapsedMs / 1000
        if (elapsedSec > ROITOOL_TIMEOUT_SEC) { timedOut = true; break }
        if (elapsedSec > 5 && fatalSeen.get()) break
    }
    if (proc.isAlive()) proc.destroyForcibly().waitFor(5, TimeUnit.SECONDS)
    reader.join(2000)
    [exitCode: proc.isAlive() ? -1 : proc.exitValue(), output: output.toString(),
     timedOut: timedOut, elapsedSec: (long)((System.currentTimeMillis() - startMs) / 1000),
     iceHost: iceHost]
}

// null on success, else "timeout", "ice-unreachable" (refused, or a reachable host
// that is not a Blitz server, e.g. the web host: "null object proxy"), "auth"
// (rejected or expired key/password) or "failed".
def roitoolFailure = { Map result ->
    if (result.timedOut) return "timeout"
    if (result.exitCode == 0) return null
    def out = result.output.toLowerCase()
    if (out.contains("connectionrefused") || out.contains("connection refused") ||
            out.contains("null object proxy")) return "ice-unreachable"
    if (out.contains("permission") || out.contains("glacier") || out.contains("password check"))
        return "auth"
    return "failed"
}

// Interactive: on an unreachable ICE host, ask for the right one and retry once (ADR-015).
// The returned result's iceHost is the host actually used.
def runWithIceRetry = { String subcommand, String iceHost, List authArgs, String stdinText ->
    def result = runRoitool(subcommand, iceHost, authArgs, stdinText)
    if (roitoolFailure(result) != "ice-unreachable") return result
    def altHost = Dialogs.showInputDialog("ICE host unreachable",
        "Could not connect to ${iceHost}:${OMERO_PORT}.\n" +
        "Enter the correct ICE/Blitz hostname to retry\n" +
        "(set OMERO_ICE_HOST in the script header to avoid this prompt):",
        iceHost)
    if (!altHost?.trim()) return result
    print("Retrying with ICE host: ${altHost.trim()}")
    return runRoitool(subcommand, altHost.trim(), authArgs, stdinText)
}

// Interactive: explains a failed run. A rejected session key clears only the key,
// keeping the ICE host, so a split web/ICE server is not re-prompted (ADR-015).
def reportRoitoolFailure = { String failure, Map result, String action, String user, boolean usedKey ->
    if (failure == "timeout") {
        Dialogs.showErrorMessage("${action} timed out",
            "roitool was still running after ${result.elapsedSec}s and was stopped.\n" +
            (action == "Upload" ? "Some ROIs may already be on OMERO — check before re-running.\n" : "") +
            "\nRaise ROITOOL_TIMEOUT_SEC (${ROITOOL_TIMEOUT_SEC}s) at the top of the script and run again.")
    } else if (failure == "auth" && usedKey) {
        saveCache(result.iceHost, user, "")
        Dialogs.showWarningNotification("Session expired",
            "Session expired — cache cleared. Please run the script again to re-authenticate.")
    } else {
        Dialogs.showErrorMessage("${action} failed",
            "roitool exited with code ${result.exitCode}.\n\n${result.output.take(500)}")
    }
}
// ==== SHARED END roitool-runner ====

// ── Step 2: Extract server + image ID ─────────────────────────────────────────
// Works whether the image was opened from OMERO or from a local file.

// ==== SHARED BEGIN omero-url — keep identical in push/pull scripts (check_shared.py) ====
def parseOmeroUrl = { String urlStr ->
    def result = [host: null, baseUrl: null, imageId: null]
    if (!urlStr) return result
    try {
        def u = new URI(urlStr)
        if (u.host) {
            result.host    = u.host
            result.baseUrl = u.scheme + "://" + u.host +
                (u.port > 0 && u.port != 443 && u.port != 80 ? ":${u.port}" : "")
        }
        // Search query string for image ID (webclient, iViewer)
        for (pattern in [[/[?&]show=image-(\d+)/, urlStr],
                         [/img_detail[\/=](\d+)/,  urlStr],
                         [/\/image[\/=](\d+)/,      urlStr],
                         [/[?&]images=(\d+)/,       urlStr]]) {
            def m = urlStr =~ pattern[0]
            if (m.find()) { result.imageId = m.group(1); break }
        }
        // PathViewer: image ID is in the fragment as "slide=<id>"
        if (!result.imageId && u.fragment) {
            def m = u.fragment =~ /slide=(\d+)/
            if (m.find()) result.imageId = m.group(1)
        }
    } catch (Exception ignored) {}
    return result
}
// ==== SHARED END omero-url ====


def server      = QP.getCurrentServer()
def imageUriStr = server?.getURIs()?.first()?.toString() ?: ""
def parsed      = parseOmeroUrl(imageUriStr)
def omeroHost    = parsed.host
def omeroBaseUrl = parsed.baseUrl
def imageId      = parsed.imageId

if (!omeroHost || !imageId) {
    def urlField  = new TextField(imageUriStr)
    urlField.setPrefWidth(440)
    urlField.setPromptText("https://omero.server.com/pathviewer/viewer/#?slide=2161771")
    def hostField = new TextField(omeroHost ?: "")
    def idField   = new TextField(imageId   ?: "")
    def vbox = new VBox(6,
        new Label("OMERO image URL (paste to auto-fill host + image ID):"), urlField,
        new Label("OMERO web host (override, e.g. omero.hms.harvard.edu — not the ICE host):"), hostField,
        new Label("Image ID (override):"),           idField)
    vbox.setPadding(new Insets(10))
    if (!Dialogs.showConfirmDialog("OMERO image details", vbox)) return

    // Parse the URL field first, then let manual fields override
    if (urlField.text.trim()) {
        def p = parseOmeroUrl(urlField.text.trim())
        if (p.host)    { omeroHost = p.host; omeroBaseUrl = p.baseUrl }
        if (p.imageId) { imageId   = p.imageId }
    }
    if (hostField.text.trim()) {
        omeroHost    = hostField.text.trim()
        omeroBaseUrl = "https://" + omeroHost
    }
    if (idField.text.trim()) imageId = idField.text.trim()
}

if (!omeroHost || !imageId) {
    Dialogs.showErrorMessage("Missing info", "Server host and Image ID are required.")
    return
}

// ── Step 3: Credentials ───────────────────────────────────────────────────────
// ==== SHARED BEGIN credentials — keep identical in push/pull scripts (check_shared.py) ====
def omeroIceHost = OMERO_ICE_HOST ?: (cachedProps.getProperty("ice_host", "") ?: omeroHost)
print("Web: ${omeroBaseUrl}  ICE: ${omeroIceHost}:${OMERO_PORT}  Image ID: ${imageId}")

def omeroUser = cachedProps.getProperty("username", "") ?: null
def omeroKey  = cachedProps.getProperty("session_key", "") ?: null
def omeroPass = null
// Note: ICE session keys cannot be validated here. The web API authenticates via
// Django session cookies; the ICE session UUID is a separate auth system on port 4064.
// There is no web endpoint that accepts an ICE UUID to check liveness — that would
// require the Blitz protocol itself, which is what roitool does. We proceed with the
// cached key and handle expiry in step 5 (roitool auth failure → clear cache → re-run).
//
// Importantly, the two sessions expire on independent clocks:
//   - ICE session: idle timeout controlled by omero.sessions.timeout (default 10 min)
//   - OMERO.web Django session: controlled by omero.web.session_cookie_age (default 24 h)
// A user can be fully logged in to the web UI while their ICE session UUID has already
// expired — hence the cached key may be stale even mid-browser-session.
if (omeroKey) print("Using cached session key for user: ${omeroUser}")

// Prompt if no valid key, then exchange username+password for an ICE session key
// via the OMERO JSON API login endpoint — so we never need to cache the password
if (!omeroKey) {
    def userField = new TextField(omeroUser ?: System.getProperty("user.name") ?: "")
    def passField = new PasswordField()
    def credBox = new VBox(6,
        new Label("OMERO username:"), userField,
        new Label("OMERO password:"), passField,
        new Label("(session key will be obtained and cached — password not stored)"))
    credBox.setPadding(new Insets(10))
    if (!Dialogs.showConfirmDialog("OMERO credentials", credBox)) return
    omeroUser = userField.text.trim()
    omeroPass = passField.text
    if (!omeroUser || !omeroPass) {
        Dialogs.showErrorMessage("Missing credentials", "Username and password are required.")
        return
    }

    // Exchange username+password for an ICE session key via /api/v0/login/
    // The returned eventContext.sessionUuid is the ICE session key usable with --key
    try {
        def trustAll = [new javax.net.ssl.X509TrustManager() {
            java.security.cert.X509Certificate[] getAcceptedIssuers() { [] as java.security.cert.X509Certificate[] }
            void checkClientTrusted(java.security.cert.X509Certificate[] c, String a) {}
            void checkServerTrusted(java.security.cert.X509Certificate[] c, String a) {}
        }] as javax.net.ssl.TrustManager[]
        def sslCtx = javax.net.ssl.SSLContext.getInstance("TLS")
        sslCtx.init(null, trustAll, new java.security.SecureRandom())
        def httpClient = java.net.http.HttpClient.newBuilder().sslContext(sslCtx)
            .connectTimeout(java.time.Duration.ofSeconds(15)).build()

        // Step 1: get CSRF token
        def tokenResp = httpClient.send(
            java.net.http.HttpRequest.newBuilder()
                .uri(java.net.URI.create("${omeroBaseUrl}/api/v0/token/"))
                .GET().header("Accept", "application/json").build(),
            java.net.http.HttpResponse.BodyHandlers.ofString())
        def csrfToken = (tokenResp.body() =~ /"data"\s*:\s*"([^"]+)"/)?.with { it.find() ? it.group(1) : null }
        def cookies   = tokenResp.headers().allValues("set-cookie").collect { it.split(";")[0] }.join("; ")

        // Step 2: POST login
        def loginBody = "username=${URLEncoder.encode(omeroUser, 'UTF-8')}" +
                        "&password=${URLEncoder.encode(omeroPass, 'UTF-8')}&server=1"
        def loginResp = httpClient.send(
            java.net.http.HttpRequest.newBuilder()
                .uri(java.net.URI.create("${omeroBaseUrl}/api/v0/login/"))
                .POST(java.net.http.HttpRequest.BodyPublishers.ofString(loginBody))
                .header("Content-Type", "application/x-www-form-urlencoded")
                .header("X-CSRFToken", csrfToken ?: "")
                .header("Cookie", cookies)
                .header("Referer", "${omeroBaseUrl}/api/v0/login/")
                .build(),
            java.net.http.HttpResponse.BodyHandlers.ofString())

        if (loginResp.statusCode() == 200) {
            // Parse sessionUuid from response: {"eventContext": {"sessionUuid": "xxxx-..."}}
            def sessionMatch = (loginResp.body() =~ /"sessionUuid"\s*:\s*"([a-f0-9\-]{36})"/)
            if (sessionMatch.find()) {
                omeroKey = sessionMatch.group(1)
                print("Obtained ICE session key via web login: ${omeroKey}")
                // Cache immediately so the next run skips the password prompt
                saveCache(omeroIceHost, omeroUser, omeroKey)
                print("Session key cached.")
            } else {
                print("Login succeeded but could not parse sessionUuid — will use password for this run.")
                print("Response: ${loginResp.body().take(200)}")
            }
        } else {
            Dialogs.showErrorMessage("Login failed",
                "Server returned HTTP ${loginResp.statusCode()}\n${loginResp.body().take(200)}")
            return
        }
    } catch (Exception e) {
        // Loud, not just printed: without a key nothing is cached, every run prompts
        // again, and headless pull runs cannot work. The usual cause is a host
        // override pointing at the ICE server, which does not serve the web API.
        def why = "${e.class.simpleName}${e.message ? ': ' + e.message : ''}"
        print("Web login to ${omeroBaseUrl} failed (${why}) — falling back to password auth with roitool.")
        Dialogs.showWarningNotification("No session key",
            "Web login to ${omeroBaseUrl} failed (${why}).\n" +
            "Continuing with password; no session key will be cached.\n" +
            "The host field must be the OMERO *web* host, not the ICE server.")
    }

    // If we got a key, save it; otherwise roitool will do its own auth with --password
    if (!omeroKey) {
        // Store username so next prompt pre-fills it, but no key
        saveCache(omeroIceHost, omeroUser, "")
    }
}
// ==== SHARED END credentials ====

// ── Step 4: Export ROIs to a temp OME-XML file ────────────────────────────────
// Scripts run on a background thread; FileChooser must be shown on the FX thread.
def showFilePicker = { String title ->
    def picked = null
    def fl = new java.util.concurrent.CountDownLatch(1)
    Platform.runLater {
        def chooser = new FileChooser()
        chooser.setTitle(title)
        chooser.getExtensionFilters().add(
            new FileChooser.ExtensionFilter("GeoJSON files", "*.geojson", "*.geojson.gz", "*.json", "*.json.gz"))
        picked = chooser.showOpenDialog(null)
        fl.countDown()
    }
    fl.await()
    picked
}

def currentRois = (QP.getDetectionObjects() + QP.getAnnotationObjects()) as List<PathROIObject>
List<PathROIObject> rois

if (currentRois.isEmpty()) {
    // No objects in image — must load from GeoJSON
    def picked = showFilePicker("No objects in image — select a GeoJSON file to upload")
    if (!picked) return
    try {
        rois = PathIO.readObjects(picked) as List<PathROIObject>
        print("Loaded ${rois.size()} object(s) from: ${picked.name}")
    } catch (Exception e) {
        Dialogs.showErrorMessage("GeoJSON load failed", "Could not read ${picked.name}:\n${e.message}")
        return
    }
} else {
    // Objects exist — default to current image, offer GeoJSON override
    def useGeoJson = new CheckBox("Load from GeoJSON file instead")
    useGeoJson.setSelected(false)
    def srcBox = new VBox(6,
        new Label("Found ${currentRois.size()} object(s) in the current image."),
        useGeoJson)
    srcBox.setPadding(new Insets(10))
    if (!Dialogs.showConfirmDialog("ROI source", srcBox)) return

    if (useGeoJson.isSelected()) {
        def picked = showFilePicker("Select a GeoJSON file to upload")
        if (!picked) return
        try {
            rois = PathIO.readObjects(picked) as List<PathROIObject>
            print("Loaded ${rois.size()} object(s) from: ${picked.name}")
        } catch (Exception e) {
            Dialogs.showErrorMessage("GeoJSON load failed", "Could not read ${picked.name}:\n${e.message}")
            return
        }
    } else {
        rois = currentRois
    }
}

if (rois.isEmpty()) {
    Dialogs.showWarningNotification("No ROIs", "No annotations or detections to export.")
    return
}

def tempXml = File.createTempFile("qupath_rois_", ".ome.xml")
tempXml.deleteOnExit()
print("Exporting ${rois.size()} object(s) to temp file: ${tempXml}")

try {
    def ome = new OME()
    def structuredAnnotations = new StructuredAnnotations()

    // nextROI doubles as both the running index and the total ROI count,
    // since addROI returns the number of ROIs it emitted (1 normally, 2 when holes
    // are split into a separate Exclude ROI, HOLE_MODE "exclude").
    int nextROI = 0
    rois.each { PathROIObject path ->
        nextROI += addROI(path.getROI(), path, nextROI, ome, structuredAnnotations)
        if (path.isCell() && path.hasNucleus()) {
            nextROI += addROI(path.getNucleusROI(), path, nextROI, ome, structuredAnnotations)
        }
    }
    ome.setStructuredAnnotations(structuredAnnotations)
    new XMLWriter().writeFile(tempXml, ome, false)
    print("Exported ${nextROI} ROI(s) to OME-XML.")
} catch (Exception e) {
    Dialogs.showErrorMessage("Export failed", "Could not export ROIs:\n${e.message}")
    tempXml.delete(); return
}

// ── Step 5: Upload via roitool subprocess ─────────────────────────────────────
print("Uploading ROIs to OMERO...")

if (!omeroKey && !omeroPass) {
    Dialogs.showErrorMessage("Authentication error", "No session key or password available — cannot upload.")
    tempXml.delete()
    return
}

try {
    def result = runWithIceRetry("import", omeroIceHost,
        roitoolAuthArgs(omeroKey, omeroUser, imageId, tempXml.absolutePath), omeroKey ? null : omeroPass)
    def failure = roitoolFailure(result)
    if (failure) {
        reportRoitoolFailure(failure, result, "Upload", omeroUser, omeroKey != null)
    } else {
        saveCache(result.iceHost, omeroUser, omeroKey)
        if (omeroKey) print("Session key cached — no password needed next run.")
        Dialogs.showInfoNotification("Success", "ROIs uploaded to OMERO image ${imageId}.")
    }
} catch (Exception e) {
    Dialogs.showErrorMessage("Execution error", "Failed to run ome-omero-roitool:\n${e.message}")
} finally {
    tempXml.delete()
}

// ── OME-XML construction helpers ──────────────────────────────────────────────

// Returns the number of OME ROIs emitted (1 for most types; 2 for geometry with
// holes under HOLE_MODE "exclude", where interior rings become a separate "Exclude" ROI).
int addROI(qupath.lib.roi.interfaces.ROI roi, PathROIObject path, int index,
            OME ome, StructuredAnnotations structuredAnnotations) {
    def mapAnnotationID = "MapAnnotation-${index}"
    def shapeID         = "Shape:${index}:0"
    def roiID           = "ROI-${index}"

    def omeROI = new ROI()
    omeROI.setID(roiID)
    // Both available: Shape.text ← name, ROI.name ← class. Only one: use it for both.
    def hasName  = path.getName() != null
    def hasClass = path.pathClass != null
    def shapeText = hasName ? path.getName() : (hasClass ? path.pathClass.name : null)
    def roiName   = hasClass ? path.pathClass.name : (hasName ? path.getName() : null)
    if (roiName) omeROI.setName(roiName)

    def mapAnnotation = new MapAnnotation()
    mapAnnotation.setID(mapAnnotationID)
    def pairList = new ArrayList<MapPair>()
    if (path.pathClass != null) pairList.add(new MapPair("qupath:class", path.pathClass.name))
    if (path.getName()) pairList.add(new MapPair("qupath:name", path.getName()))
    else if (path.isCell()) {
        pairList.add(new MapPair("qupath:name",
            roi.equals(path.getNucleusROI()) ? "cell nucleus" : "cell boundary"))
    }
    pairList.add(new MapPair("qupath:is-annotation", path.isAnnotation().toString()))
    pairList.add(new MapPair("qupath:is-detection",  path.isDetection().toString()))
    path.retrieveMetadataKeys().each {
        pairList.add(new MapPair("qupath:metadata:" + it, path.retrieveMetadataValue(it).toString()))
    }
    mapAnnotation.setValue(pairList)

    def union = new Union()

    switch (roi) {
        case EllipseROI:
            def e = roi as EllipseROI
            def shape = new Ellipse()
            shape.setID(shapeID); shape.setX(e.getCentroidX()); shape.setY(e.getCentroidY())
            shape.setRadiusX(e.getBoundsWidth() / 2); shape.setRadiusY(e.getBoundsHeight() / 2)
            setShapeText(shape, shapeText); setCommonProps(shape, path, roi)
            union.addShape(shape as Shape); break

        case LineROI:
            def l = roi as LineROI
            def shape = new Line()
            shape.setID(shapeID); shape.setX1(l.x1); shape.setY1(l.y1)
            shape.setX2(l.x2); shape.setY2(l.y2)
            setShapeText(shape, shapeText); setCommonProps(shape, path, roi)
            union.addShape(shape as Shape); break

        case PointsROI:
            (roi as PointsROI).getPointList().eachWithIndex { pt, i ->
                def shape = new Point()
                shape.setID(shapeID + "." + i); shape.setX(pt.x); shape.setY(pt.y)
                setShapeText(shape, shapeText); setCommonProps(shape, path, roi)
                union.addShape(shape)
            }; break

        case PolygonROI:
            def shape = new Polygon()
            shape.setID(shapeID)
            shape.setPoints(roi.getAllPoints().collect { String.format("%f,%f", it.X, it.Y) }.join(" "))
            setShapeText(shape, shapeText); setCommonProps(shape, path, roi)
            union.addShape(shape); break

        case PolylineROI:
            def shape = new Polyline()
            shape.setID(shapeID)
            shape.setPoints((roi as PolylineROI).getAllPoints().collect { String.format("%f,%f", it.X, it.Y) }.join(" "))
            setShapeText(shape, shapeText); setCommonProps(shape, path, roi)
            union.addShape(shape as Shape); break

        case RectangleROI:
            def r = roi as RectangleROI
            def shape = new Rectangle()
            shape.setID(shapeID); shape.setX(r.x); shape.setY(r.y)
            shape.setWidth(r.getBoundsWidth()); shape.setHeight(r.getBoundsHeight())
            setShapeText(shape, shapeText); setCommonProps(shape, path, roi)
            union.addShape(shape as Shape); break

        case AreaROI:
        case GeometryROI:
            def jtsGeom = roi.getGeometry()
            def polys = []
            if (jtsGeom.geometryType in ["MultiPolygon", "GeometryCollection"]) {
                for (int gi = 0; gi < jtsGeom.numGeometries; gi++) polys << jtsGeom.getGeometryN(gi)
            } else { polys << jtsGeom }

            // "union": holes ride along as ".hole." shapes of this ROI. "exclude": they
            // go to a separate Exclude ROI that Pageant subtracts (ADR-017).
            def holeUnion = HOLE_MODE == "exclude" ? new Union() : union

            polys.eachWithIndex { poly, pi ->
                def coords = poly.exteriorRing?.coordinates ?: poly.coordinates
                if (!coords) return
                def shape = new Polygon()
                shape.setID(pi == 0 ? shapeID : "${shapeID}.${pi}")
                shape.setPoints(coords.collect { String.format("%f,%f", it.x, it.y) }.join(" "))
                setShapeText(shape, shapeText); setCommonProps(shape, path, roi)
                union.addShape(shape)
                if (poly.respondsTo("getNumInteriorRing")) {
                    for (int hi = 0; hi < poly.numInteriorRing; hi++) {
                        def holeShape = new Polygon()
                        holeShape.setID("${shapeID}.hole.${pi}.${hi}")
                        holeShape.setPoints(poly.getInteriorRingN(hi).coordinates
                            .collect { String.format("%f,%f", it.x, it.y) }.join(" "))
                        setShapeText(holeShape, shapeText); setCommonProps(holeShape, path, roi)
                        holeUnion.addShape(holeShape)
                    }
                }
            }

            if (holeUnion.is(union) || holeUnion.sizeOfShapeList() == 0) break

            // Emit the exterior ROI, then the Exclude ROI for its interior rings
            omeROI.setUnion(union)
            omeROI.linkAnnotation(mapAnnotation)
            ome.addROI(omeROI)
            structuredAnnotations.addMapAnnotation(mapAnnotation)

            def excludeROI = new ROI()
            excludeROI.setID("ROI-${index}-exclude")
            excludeROI.setName("Exclude")
            def excludeAnnotation = new MapAnnotation()
            excludeAnnotation.setID("MapAnnotation-${index}-exclude")
            excludeAnnotation.setValue(new ArrayList<MapPair>(pairList))
            excludeROI.setUnion(holeUnion)
            excludeROI.linkAnnotation(excludeAnnotation)
            ome.addROI(excludeROI)
            structuredAnnotations.addMapAnnotation(excludeAnnotation)
            return 2

        default:
            print("Unsupported ROI type: ${roi.class.simpleName} — skipped")
    }

    if (union.sizeOfShapeList() > 0) {
        omeROI.setUnion(union)
        omeROI.linkAnnotation(mapAnnotation)
        ome.addROI(omeROI)
        structuredAnnotations.addMapAnnotation(mapAnnotation)
    }
    return 1
}

static void setShapeText(Shape shape, String text) {
    if (text) shape.setText(text)
}

static void setCommonProps(Shape shape, PathROIObject path,
                            qupath.lib.roi.interfaces.ROI roi) {
    if (roi.c > -1) shape.setTheC(new NonNegativeInteger(roi.c))
    shape.setTheT(new NonNegativeInteger(roi.t))
    shape.setTheZ(new NonNegativeInteger(roi.z))
    shape.setLocked(path.isLocked())
    shape.setFillRule(FillRule.NONZERO)
    def packedColor = path.getColor() ?: path.pathClass?.getColor() ?: PathPrefs.colorDefaultObjectsProperty().get()
    if (packedColor != null) {
        shape.setStrokeColor(new Color(ColorTools.red(packedColor), ColorTools.green(packedColor),
                                       ColorTools.blue(packedColor), ColorTools.alpha(packedColor)))
    }
    // Fill color skipped intentionally — would require getCurrentViewer().getOverlayOptions(), not headless-safe
    if (path.isAnnotation())
        shape.setStrokeWidth(new Length(PathPrefs.annotationStrokeThicknessProperty().get(), UNITS.PIXEL))
    else if (path.isDetection())
        shape.setStrokeWidth(new Length(PathPrefs.detectionStrokeThicknessProperty().get(), UNITS.PIXEL))
}
