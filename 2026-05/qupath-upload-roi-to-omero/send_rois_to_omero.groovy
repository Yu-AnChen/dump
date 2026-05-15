/**
 * send_rois_to_omero.groovy
 *
 * Sends QuPath ROIs to OMERO via ome-omero-roitool.
 *
 * What this script does:
 *   1. Downloads and caches ome-omero-roitool (once, ~77MB)
 *   2. Downloads and caches a JRE from Adoptium (once, ~50MB) — no manual Java install needed
 *   3. Extracts OMERO server host and image ID from the currently open image URI
 *   4. Resolves credentials: cached ICE session key → password prompt → web login for session key
 *   5. Exports all annotations + detections to a temp OME-XML file
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
ROITOOL_VERSION  = "0.2.7"
OMERO_PORT       = 4064
JRE_VERSION      = "21"
// Optional: set if your OMERO Blitz/ICE server has a different hostname than
// the web frontend (e.g. omero-blitz.server.com vs omero-web.server.com).
// Leave as "" to use the hostname parsed from the image URL.
OMERO_ICE_HOST   = ""
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

// ── Step 0: Resolve roitool binary ────────────────────────────────────────────
def toolDir = new File(userDir, "ome-omero-roitool-${ROITOOL_VERSION}")
def toolZip = new File(userDir, "ome-omero-roitool-${ROITOOL_VERSION}.zip")
def exeName = isWindows ? "ome-omero-roitool.bat" : "ome-omero-roitool"

def findExe = { File dir ->
    def found = null
    if (dir.exists()) dir.eachFileRecurse { f -> if (!f.directory && f.name == exeName) found = f }
    return found
}

def toolExe = findExe(toolDir)

if (toolExe == null) {
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
        toolExe = findExe(toolDir)
        if (toolExe == null) {
            Dialogs.showErrorMessage("Install failed", "Could not find '${exeName}' after extracting.")
            return
        }
        if (!isWindows) toolExe.setExecutable(true)
        print("Downloaded roitool: ${toolExe.absolutePath}")
    } catch (Exception e) {
        Dialogs.showErrorMessage("Download failed", "Could not download ome-omero-roitool:\n${e.message}")
        return
    }
} else {
    print("Using cached roitool: ${toolExe.absolutePath}")
    if (!isWindows && !toolExe.canExecute()) toolExe.setExecutable(true)
}

// ── Step 1: Resolve Java (auto-downloaded Adoptium JRE) ───────────────────────
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

// ── Step 2: Extract server + image ID ─────────────────────────────────────────
// Works whether the image was opened from OMERO or from a local file.

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
        new Label("OMERO server host (override):"), hostField,
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
// Tier 1: cached ICE session key → Tier 2: password prompt
def credsFile   = new File(userDir, ".omero_qupath_creds")
def cachedProps = new Properties()
if (credsFile.exists()) credsFile.withInputStream { cachedProps.load(it) }

def omeroIceHost = OMERO_ICE_HOST ?: (cachedProps.getProperty("ice_host", "") ?: omeroHost)
print("Web: ${omeroBaseUrl}  ICE: ${omeroIceHost}:${OMERO_PORT}  Image ID: ${imageId}")

def saveCache = { String iceHost, String user, String key ->
    def props = new Properties()
    props.setProperty("ice_host",    iceHost ?: "")
    props.setProperty("username",    user ?: "")
    props.setProperty("session_key", key ?: "")
    credsFile.withOutputStream { props.store(it, "QuPath OMERO session cache — delete to reset") }
}

def omeroKey  = null
def omeroUser = null
def omeroPass = null

omeroUser = cachedProps.getProperty("username", "") ?: null
omeroKey  = cachedProps.getProperty("session_key", "") ?: null
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
        def httpClient = java.net.http.HttpClient.newBuilder().sslContext(sslCtx).build()

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
        print("Web login failed (${e.message}) — falling back to password auth with roitool.")
    }

    // If we got a key, save it; otherwise roitool will do its own auth with --password
    if (!omeroKey) {
        // Store username so next prompt pre-fills it, but no key
        saveCache(omeroIceHost, omeroUser, "")
    }
}

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

    int nextROI = 0
    rois.each { PathROIObject path ->
        addROI(path.getROI(), path, nextROI, ome, structuredAnnotations)
        nextROI++
        if (path.isCell() && path.hasNucleus()) {
            addROI(path.getNucleusROI(), path, nextROI, ome, structuredAnnotations)
            nextROI++
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
// Roitool runs as a subprocess so ICE's System.exit() kills only the child,
// leaving QuPath alive.
print("Uploading ROIs to OMERO...")

// Use --key if we have a session key (from cache or web login), else --username/--password
if (!omeroKey && !omeroPass) {
    Dialogs.showErrorMessage("Authentication error", "No session key or password available — cannot upload.")
    tempXml.delete()
    return
}

def roitoolArgs = omeroKey ?
    ["--key", omeroKey, imageId, tempXml.absolutePath] :
    ["--username", omeroUser, "--password", omeroPass, imageId, tempXml.absolutePath]

def runRoitool = { String iceHost ->
    def importArgs = ["import", "--server", iceHost, "--port", OMERO_PORT.toString()] + roitoolArgs
    def cmd
    def cpArgFile = null

    if (isWindows) {
        // Windows cmd.exe has an 8191-char line limit; the roitool .bat hits this when
        // expanding its classpath inline. Parse the classpath from the .bat directly
        // and invoke java.exe via a @argfile to keep the command line short.
        def appHome = toolExe.parentFile.parentFile.absolutePath
        def classpath = null
        def mainClass = "com.glencoesoftware.roitool.Main"
        toolExe.eachLine { line ->
            def t = line.trim()
            if (t.startsWith("set CLASSPATH="))
                classpath = t.substring("set CLASSPATH=".length()).replace("%APP_HOME%", appHome)
            if (t.contains("-classpath") && t.contains("%*")) {
                def m = t =~ /\s([a-z][a-zA-Z0-9.]+)\s+%\*/
                if (m.find()) mainClass = m.group(1)
            }
        }
        if (!classpath) {
            // Fallback: scan lib/ directly
            def jars = []
            new File(appHome, "lib").eachFileRecurse { f ->
                if (!f.directory && f.name.endsWith(".jar")) jars << f.absolutePath
            }
            classpath = jars.join(';')
        }
        // Forward slashes avoid backslash-escape issues inside @argfile quoted strings
        cpArgFile = File.createTempFile("roitool_cp_", ".args")
        cpArgFile.deleteOnExit()
        cpArgFile.text = "-cp\n\"${classpath.replace('\\', '/')}\"\n"
        cmd = [javaHome.toString() + "\\bin\\java.exe",
               "@" + cpArgFile.absolutePath,
               mainClass] + importArgs
    } else {
        cmd = ["/bin/bash", toolExe.absolutePath] + importArgs
    }

    def pb = new ProcessBuilder(cmd)
    pb.redirectErrorStream(true)
    pb.environment().put("JAVA_HOME", javaHome.toString())
    pb.environment().put("PATH", javaHome.toString() + (isWindows ? "\\bin;" : "/bin:") +
        (pb.environment().getOrDefault("PATH", isWindows ? "" : "/usr/bin:/bin")))
    try {
        def proc = pb.start()
        def output = new StringBuilder()
        // ICE daemon threads keep the subprocess alive after the main thread crashes.
        // Read output on a background thread, and force-kill the process 5s after
        // detecting a fatal error rather than waiting for the full 120s timeout.
        def fatalSeen = new java.util.concurrent.atomic.AtomicBoolean(false)
        def reader = Thread.start {
            proc.inputStream.eachLine { line ->
                print(line); output.append(line).append("\n")
                if (line.contains("ConnectionRefusedException") ||
                        line.contains("PermissionDeniedException") ||
                        line.contains("Connection refused"))
                    fatalSeen.set(true)
            }
        }
        long startMs = System.currentTimeMillis()
        while (proc.isAlive()) {
            if (proc.waitFor(500, TimeUnit.MILLISECONDS)) break
            long elapsedSec = (System.currentTimeMillis() - startMs) / 1000
            if (elapsedSec > 120) break
            if (elapsedSec > 5 && fatalSeen.get()) break
        }
        if (proc.isAlive()) proc.destroyForcibly().waitFor(5, TimeUnit.SECONDS)
        reader.join(2000) // 2s drain
        [exitCode: proc.isAlive() ? -1 : proc.exitValue(), output: output.toString()]
    } finally {
        cpArgFile?.delete()
    }
}

try {
    def effectiveIceHost = omeroIceHost
    def result = runRoitool(effectiveIceHost)

    // If connection refused, offer to retry with a different ICE hostname
    if (result.output.toLowerCase().contains("connectionrefused") ||
            result.output.toLowerCase().contains("connection refused")) {
        def altHost = Dialogs.showInputDialog("ICE host unreachable",
            "Could not connect to ${effectiveIceHost}:${OMERO_PORT}.\n" +
            "Enter the correct ICE/Blitz hostname to retry\n" +
            "(set OMERO_ICE_HOST in the script header to avoid this prompt):",
            effectiveIceHost)
        if (!altHost?.trim()) {
            Dialogs.showErrorMessage("Upload failed",
                "roitool exited with code ${result.exitCode}.\n\n${result.output.take(500)}")
            return
        }
        effectiveIceHost = altHost.trim()
        print("Retrying with ICE host: ${effectiveIceHost}")
        result = runRoitool(effectiveIceHost)
    }

    int exitCode = result.exitCode
    def out = result.output

    if (exitCode == 0) {
        saveCache(effectiveIceHost, omeroUser, omeroKey)
        if (omeroKey) print("Session key cached — no password needed next run.")
        Dialogs.showInfoNotification("Success", "ROIs uploaded to OMERO image ${imageId}.")
    } else {
        def outLower = out.toLowerCase()
        // Heuristic: ICE auth errors mention "permission", "glacier", or "password check".
        // If roitool changes its error messages, this detection silently falls through to the
        // generic error dialog — clear the cache manually in that case.
        def authFailure = omeroKey && (outLower.contains("permission") ||
            outLower.contains("glacier") || outLower.contains("password check"))
        if (authFailure) {
            // Clear stale key and ask user to re-run
            credsFile.delete()
            Dialogs.showWarningNotification("Session expired",
                "Session expired — cache cleared. Please run the script again to re-authenticate.")
        } else {
            Dialogs.showErrorMessage("Upload failed",
                "roitool exited with code ${exitCode}.\n\n${out.take(500)}")
        }
    }
} catch (Exception e) {
    Dialogs.showErrorMessage("Execution error", "Failed to run ome-omero-roitool:\n${e.message}")
} finally {
    tempXml.delete()
}

// ── OME-XML construction helpers ──────────────────────────────────────────────

void addROI(qupath.lib.roi.interfaces.ROI roi, PathROIObject path, int index,
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
                        union.addShape(holeShape)
                    }
                }
            }; break

        default:
            print("Unsupported ROI type: ${roi.class.simpleName} — skipped")
    }

    if (union.sizeOfShapeList() > 0) {
        omeROI.setUnion(union)
        omeROI.linkAnnotation(mapAnnotation)
        ome.addROI(omeROI)
        structuredAnnotations.addMapAnnotation(mapAnnotation)
    }
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
