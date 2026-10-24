/**
 * pull_rois_from_omero.groovy
 *
 * Reverse of send_rois_to_omero.groovy: downloads ROIs from an OMERO
 * image and adds them to the current QuPath image as annotations/detections.
 *
 * Auth, host resolution, JRE provisioning and subprocess handling are the SHARED
 * blocks of send_rois_to_omero.groovy, kept identical by check_shared.py (ADR-023).
 * Only the roitool sub-command changes: "export" instead of "import".
 *
 * What this script does:
 *   1. Downloads and caches ome-omero-roitool (once, ~77MB)      — shared cache with the push script
 *   2. Downloads and caches a JRE from Adoptium (once, ~50MB)
 *   3. Extracts OMERO server host and image ID from the current image URI
 *   4. Resolves credentials: cached ICE session key → password prompt → web login
 *   5. Runs `ome-omero-roitool export` as a subprocess into a temp OME-XML file
 *   6. Converts the OME-XML back into QuPath objects:
 *        - all polygon shapes of one OME ROI → ONE QuPath object (restores MultiPolygon)
 *        - ROIs named "Exclude" are spatially subtracted from their container,
 *          inverting the Pageant hole convention (ADR-017)
 *        - MapAnnotation qupath:* pairs restore class, name, and metadata
 *        - each shape's Transform is applied, e.g. PathViewer's rotated shapes (ADR-025)
 *   7. Cleans up the temp file
 *
 * Requirements:
 *   - QuPath 0.7 (uses PathClass.fromString / GeometryTools.geometryToROI)
 *   - An image open in QuPath to receive the objects
 *   - OMERO ICE port 4064 reachable
 *
 * Offline / testing:
 *   --file <path.ome.xml>   skip OMERO entirely and import a local OME-XML (no image needed)
 *   --geojson <path>        with --file or --id: also write the objects as a GeoJSON FeatureCollection
 *
 * Headless batch (no dialogs; needs a session key cached by one interactive run):
 *   --id <image id> --xml <out.ome.xml> [--geojson <out.geojson>]   (ADR-027)
 *   --selftest              run the conversion checks, no network or image needed
 *   (pass via QuPath CLI:  QuPath script pull_rois_from_omero.groovy -a --selftest)
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

// ROI names that carry no classification meaning. PathViewer names every drawn shape
// "Annotation", which would otherwise become a QuPath class on every object pulled.
// Names not listed here (e.g. Pageant's "Include") still become the object's class.
GENERIC_ROI_NAMES = ["Annotation"]
// ──────────────────────────────────────────────────────────────────────────────

import java.net.URI
import java.net.URL
import java.util.concurrent.TimeUnit
import java.util.zip.*
import javafx.geometry.Insets
import javafx.scene.control.CheckBox
import javafx.scene.control.Label
import javafx.scene.control.PasswordField
import javafx.scene.control.TextField
import javafx.scene.layout.VBox
import loci.common.services.ServiceFactory
import loci.formats.services.OMEXMLService
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.util.AffineTransformation
import qupath.lib.common.ColorTools
import qupath.lib.geom.Point2
import qupath.lib.gui.prefs.PathPrefs
import qupath.lib.objects.PathObject
import qupath.lib.objects.PathObjects
import qupath.lib.objects.classes.PathClass
import qupath.lib.regions.ImagePlane
import qupath.lib.roi.GeometryTools
import qupath.lib.roi.ROIs
import qupath.lib.scripting.QP

def scriptArgs = binding.hasVariable('args') ? (args as List) : []

// ── Self-test: hole reconstruction, no OMERO or image needed ───────────────────
if (scriptArgs.contains('--selftest')) {
    def gf = GeometryTools.getDefaultFactory()
    def square = { double x0, double y0, double x1, double y1 ->
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].collect { new Coordinate(it[0], it[1]) } as Coordinate[]
    }

    // Geometry layer: an Exclude region is subtracted only where it is contained
    def outer = buildAreaGeometry([square(0, 0, 100, 100)], [], gf)
    assert outer.area == 10000d : "exterior area ${outer.area}"
    def holed = subtractExcludes(outer, [buildAreaGeometry([square(25, 25, 75, 75)], [], gf)])
    assert holed.area == 7500d            : "expected 7500 after subtraction, got ${holed.area}"
    assert holed.getNumInteriorRing() == 1 : "expected 1 interior ring, got ${holed.getNumInteriorRing()}"
    def elsewhere = buildAreaGeometry([square(500, 500, 600, 600)], [], gf)
    assert subtractExcludes(outer, [elsewhere]).area == 10000d : "non-contained Exclude must not subtract"
    // Interior rings carried inside one ROI (base-script convention) also survive
    assert buildAreaGeometry([square(0, 0, 100, 100)], [square(25, 25, 75, 75)], gf).area == 7500d

    // End-to-end: the exact OME-XML shape the Pageant push script emits, including the
    // ".hole." Shape IDs it keeps inside an Exclude ROI. Regression guard: those rings
    // are the Exclude's own exteriors, not holes of it.
    def xml = File.createTempFile("pull_selftest_", ".ome.xml")
    xml.text = '''<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <ROI ID="ROI-0" Name="Stroma">
    <Union>
      <Polygon ID="Shape:0:0" Points="0,0 100,0 100,100 0,100" TheZ="0" TheT="0" Locked="true"/>
    </Union>
    <AnnotationRef ID="MapAnnotation-0"/>
  </ROI>
  <ROI ID="ROI-1" Name="Annotation">
    <Union>
      <Point ID="Shape:1:0" X="10" Y="20" TheZ="0" TheT="0" Text="dot 1"/>
    </Union>
  </ROI>
  <ROI ID="ROI-2" Name="Annotation">
    <Union>
      <Rectangle ID="Shape:2:0" X="0" Y="0" Width="10" Height="20" TheZ="0" TheT="0" Text="rot">
        <Transform A00="0" A01="-1" A02="100" A10="1" A11="0" A12="0"/>
      </Rectangle>
    </Union>
  </ROI>
  <ROI ID="ROI-0-exclude" Name="Exclude">
    <Union>
      <Polygon ID="Shape:0:0.hole.0.0" Points="25,25 75,25 75,75 25,75" TheZ="0" TheT="0"/>
    </Union>
    <AnnotationRef ID="MapAnnotation-0-exclude"/>
  </ROI>
  <StructuredAnnotations>
    <MapAnnotation ID="MapAnnotation-0">
      <Value>
        <M K="qupath:class">Stroma</M>
        <M K="qupath:name">region A</M>
        <M K="qupath:is-annotation">true</M>
        <M K="qupath:is-detection">false</M>
        <M K="qupath:metadata:note">kept</M>
      </Value>
    </MapAnnotation>
    <MapAnnotation ID="MapAnnotation-0-exclude">
      <Value><M K="qupath:class">Stroma</M></Value>
    </MapAnnotation>
  </StructuredAnnotations>
</OME>'''
    def h = new qupath.lib.objects.hierarchy.PathObjectHierarchy()
    int n = importOmeXml(xml, h)
    xml.delete()
    assert n == 3 : "expected 3 objects (merged exterior + 1 point + 1 rotated rect), got ${n}"
    def all = h.getAllObjects(false).findAll { it.hasROI() }
    def obj = all.find { it.getName() == "region A" }
    def g = obj.getROI().getGeometry()
    assert obj.isAnnotation()               : "expected an annotation"
    assert obj.pathClass?.name == "Stroma"  : "class lost: ${obj.pathClass?.name}"
    assert obj.getName() == "region A"      : "name lost: ${obj.getName()}"
    assert obj.isLocked()                   : "locked flag lost"
    assert obj.getMetadata()["note"] == "kept" : "metadata lost: ${obj.getMetadata()}"
    assert g.area == 7500d                  : "Exclude not re-subtracted: area ${g.area}"
    assert g.getNumInteriorRing() == 1      : "expected 1 hole, got ${g.getNumInteriorRing()}"

    // A PathViewer-style ROI: generic name must NOT become a class, and the label
    // must come from the shape's Text since there is no MapAnnotation.
    def pt = all.find { it.getROI().isPoint() }
    assert pt != null            : "point object missing"
    assert pt.pathClass == null  : "generic ROI name leaked into a class: ${pt.pathClass?.name}"
    assert pt.getName() == "dot 1" : "shape Text must become the name, got ${pt.getName()}"

    // Shape Transform: x' = A00*x + A01*y + A02, y' = A10*x + A11*y + A12. A 90° turn
    // plus x+100 maps the 10x20 rect to x in [80,100], y in [0,10]; a transposed
    // matrix would land elsewhere.
    def rot = all.find { it.getName() == "rot" }
    assert rot != null : "rotated rectangle missing"
    def env = rot.getROI().getGeometry().getEnvelopeInternal()
    assert [env.minX, env.maxX, env.minY, env.maxY] == [80d, 100d, 0d, 10d] :
        "Transform not applied correctly, envelope ${env}"

    println "selftest OK"
    return
}

boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win")
def userDir = PathPrefs.userPathProperty().get() ?:
    new File(System.getProperty("user.home"), ".qupath-roitool").absolutePath
new File(userDir).mkdirs()

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

def argAfter = { String flag ->
    int i = scriptArgs.indexOf(flag)
    (i >= 0 && i + 1 < scriptArgs.size()) ? scriptArgs[i + 1] as String : null
}

// Imports an OME-XML into the open image, or a standalone hierarchy when headless,
// and writes --geojson <path> if given (coordinates are full-resolution pixels).
def importAndWrite = { File xml ->
    def target = QP.getCurrentHierarchy() ?: new qupath.lib.objects.hierarchy.PathObjectHierarchy()
    def added = importOmeXml(xml, target)
    print("Imported ${added} object(s) from ${xml.name}")
    def out = argAfter('--geojson')
    if (out) {
        QP.exportObjectsToGeoJson(target.getAllObjects(false).findAll { it.hasROI() }, out, "FEATURE_COLLECTION")
        print("Wrote ${out}")
    }
    if (target.is(QP.getCurrentHierarchy())) QP.fireHierarchyUpdate()
}

// ── Offline path: import a local OME-XML, skip everything OMERO ────────────────
if (argAfter('--file')) {
    def f = new File(argAfter('--file'))
    if (!f.exists()) throw new FileNotFoundException("No such file: ${f}")
    importAndWrite(f)
    return
}

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

// ── Headless path: --id <image id> --xml <out.ome.xml> [--geojson <out.geojson>] ─
// For batch runs (pull_rois_on_orion.bat). Never prompts: uses the ICE host and
// session key cached by an interactive run, and fails with an exception (QuPath
// exits 1) instead of a dialog. Same 5s fatal-kill and timeout as the GUI path.
if (argAfter('--id')) {
    def id      = argAfter('--id')
    def xmlPath = argAfter('--xml')
    if (!xmlPath) throw new IllegalArgumentException("--id needs --xml <path>")
    def iceHost = OMERO_ICE_HOST ?: cachedProps.getProperty("ice_host", "")
    def key     = cachedProps.getProperty("session_key", "")
    if (!iceHost || !key)
        throw new IllegalStateException("No cached ICE host/session key in ${credsFile} — " +
            "run this script once inside QuPath to log in.")

    def xml = new File(xmlPath)
    xml.delete()
    // roitool will not create the output folder, and fails only after the whole fetch
    xml.absoluteFile.parentFile?.mkdirs()
    argAfter('--geojson')?.with { new File(it).absoluteFile.parentFile?.mkdirs() }
    print("Exporting OMERO image ${id} from ${iceHost} -> ${xml}")
    def result = runRoitool("export", iceHost, roitoolAuthArgs(key, null, id, xml.absolutePath), null)
    switch (roitoolFailure(result)) {
        case null: break
        case "timeout":
            throw new IllegalStateException("roitool still running after ${result.elapsedSec}s; " +
                "raise ROITOOL_TIMEOUT_SEC")
        case "auth":
            // Clear the key: the batch checks for it and stops, instead of every later
            // image failing the same way.
            saveCache(iceHost, cachedProps.getProperty("username", ""), "")
            throw new IllegalStateException("Session key rejected (expired?) — cache cleared. " +
                "Run this script inside QuPath to log in again.")
        case "ice-unreachable":
            throw new IllegalStateException("Cannot reach ICE server ${iceHost}:${OMERO_PORT} — " +
                "run this script inside QuPath once to set the ICE host.")
        default:
            throw new IllegalStateException("roitool export failed (exit ${result.exitCode})")
    }
    if (!xml.exists()) throw new IllegalStateException("roitool reported success but wrote no ${xml}")
    importAndWrite(xml)
    return
}

def hierarchy = QP.getCurrentHierarchy()
if (hierarchy == null) {
    Dialogs.showErrorMessage("No image", "Open an image in QuPath to receive the ROIs.")
    return
}

// ── Step 2: Extract server + image ID ─────────────────────────────────────────
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

// Always confirm: the target image ID decides which ROIs land in this QuPath image,
// and a wrong ID silently imports another slide's annotations.
def urlField  = new TextField(imageUriStr)
urlField.setPrefWidth(440)
urlField.setPromptText("https://omero.server.com/pathviewer/viewer/#?slide=2161771")
def hostField = new TextField(omeroHost ?: "")
def idField   = new TextField(imageId   ?: "")
def clearFirst = new CheckBox("Delete existing objects in this image first")
def vbox = new VBox(6,
    new Label("OMERO image URL (paste to auto-fill host + image ID):"), urlField,
    new Label("OMERO web host (override, e.g. omero.hms.harvard.edu — not the ICE host):"), hostField,
    new Label("Image ID (override):"),           idField,
    clearFirst)
vbox.setPadding(new Insets(10))
if (!Dialogs.showConfirmDialog("Pull ROIs from OMERO image", vbox)) return

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

if (!omeroHost || !imageId) {
    Dialogs.showErrorMessage("Missing info", "Server host and Image ID are required.")
    return
}

// ── Step 3: Credentials (ADR-012) ─────────────────────────────────────────────
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

// ── Step 4: Export from OMERO via roitool subprocess ──────────────────────────
def tempXml = File.createTempFile("omero_rois_", ".ome.xml")
tempXml.deleteOnExit()
print("Exporting ROIs from OMERO image ${imageId}...")

if (!omeroKey && !omeroPass) {
    Dialogs.showErrorMessage("Authentication error", "No session key or password available — cannot export.")
    tempXml.delete()
    return
}

try {
    def result = runWithIceRetry("export", omeroIceHost,
        roitoolAuthArgs(omeroKey, omeroUser, imageId, tempXml.absolutePath), omeroKey ? null : omeroPass)
    def failure = roitoolFailure(result)
    if (failure) {
        reportRoitoolFailure(failure, result, "Export", omeroUser, omeroKey != null)
        return
    }
    saveCache(result.iceHost, omeroUser, omeroKey)

    // ── Step 5: OME-XML → QuPath objects ──────────────────────────────────────
    if (clearFirst.isSelected()) {
        def existing = hierarchy.getAllObjects(false)
        if (existing) {
            print("Removing ${existing.size()} existing object(s)")
            hierarchy.removeObjects(new ArrayList(existing), false)
        }
    }

    int added = importOmeXml(tempXml, hierarchy)
    QP.fireHierarchyUpdate()
    if (added == 0) {
        Dialogs.showWarningNotification("No ROIs", "OMERO image ${imageId} has no ROIs to import.")
    } else {
        Dialogs.showInfoNotification("Success", "Imported ${added} object(s) from OMERO image ${imageId}.")
    }
} catch (Exception e) {
    Dialogs.showErrorMessage("Execution error", "Failed to run ome-omero-roitool:\n${e.message}")
} finally {
    tempXml.delete()
}

// ── OME-XML → QuPath conversion ───────────────────────────────────────────────

/**
 * Reads an OME-XML file and adds the corresponding objects to the hierarchy.
 * Returns the number of objects added.
 */
int importOmeXml(File xmlFile, hierarchy) {
    def omexml = new ServiceFactory().getInstance(OMEXMLService).createOMEXMLMetadata(xmlFile.getText("UTF-8"))
    int roiCount = omexml.getROICount()
    if (roiCount < 1) { print("No ROIs found in OME-XML."); return 0 }
    print("OME-XML contains ${roiCount} ROI(s)")

    // MapAnnotation IDs are not list indices once "-exclude" ROIs are interleaved,
    // so resolve AnnotationRef → index by ID rather than by parsing the number out.
    def annIdxById = [:]
    int annCount = omexml.getMapAnnotationCount()
    for (int i = 0; i < annCount; i++) annIdxById[omexml.getMapAnnotationID(i)] = i

    def gf = GeometryTools.getDefaultFactory()
    def pending  = []   // [geometry, plane, props, pathClass, locked, color] for area ROIs
    def excludes = []   // geometries to subtract (ADR-017)
    def objects  = []

    for (int roiIdx = 0; roiIdx < roiCount; roiIdx++) {
        def props = [:]
        int refCount = omexml.getROIAnnotationRefCount(roiIdx)
        for (int r = 0; r < refCount; r++) {
            def idx = annIdxById[omexml.getROIAnnotationRef(roiIdx, r)]
            if (idx == null) continue
            omexml.getMapAnnotationValue(idx).each { props[it.name] = it.value }
        }

        def roiName = omexml.getROIName(roiIdx)
        def parsedShapes = readShapes(omexml, roiIdx, gf)

        // Interior rings live either in a separate "Exclude" ROI (Pageant) or as
        // ".hole." shapes inside this Union (base script) — handle both.
        if ("Exclude".equalsIgnoreCase(roiName)) {
            // The push script keeps ".hole." in the Shape IDs of an Exclude ROI, so
            // readShapes files them as holes — but within this ROI they are its own
            // exteriors: they describe the region to cut out.
            def rings = parsedShapes.exteriors + parsedShapes.holes
            if (rings) excludes << buildAreaGeometry(rings, [], gf)
        } else if (parsedShapes.exteriors) {
            pending << [geom: buildAreaGeometry(parsedShapes.exteriors, parsedShapes.holes, gf),
                        plane: parsedShapes.plane, props: props,
                        roiName: roiName, color: parsedShapes.color, locked: parsedShapes.locked,
                        text: parsedShapes.text]
        }

        // Non-area shapes map one-to-one onto objects; no hole semantics apply.
        parsedShapes.others.each { s ->
            objects << makeObject(s.roi, props, roiName, s.color, s.locked, s.text)
        }
    }

    // Subtract Exclude geometries from whichever object spatially contains them
    pending.each { p ->
        def geom = subtractExcludes(p.geom, excludes)
        def roi  = GeometryTools.geometryToROI(geom, p.plane)
        objects << makeObject(roi, p.props, p.roiName, p.color, p.locked, p.text)
    }

    objects.removeAll { it == null }
    if (objects) hierarchy.addObjects(objects)
    return objects.size()
}

/**
 * Pulls every shape of one OME ROI apart into area geometry rings and non-area ROIs.
 * Returns [exteriors: List<Coordinate[]>, holes: List<Coordinate[]>, others: List,
 *          plane: ImagePlane, color: Integer, locked: Boolean]
 */
Map readShapes(omexml, int roiIdx, GeometryFactory gf) {
    def out = [exteriors: [], holes: [], others: [], plane: ImagePlane.getDefaultPlane(),
               color: null, locked: null, text: null]
    int shapeCount = omexml.getShapeCount(roiIdx)

    for (int s = 0; s < shapeCount; s++) {
        def type = omexml.getShapeType(roiIdx, s)
        // Check the type before touching any getter: the accessors are per-type
        // (getPolygonTheC, ...), so an unmapped type would throw and abort the import.
        if (!(type in ["Polygon", "Rectangle", "Ellipse", "Line", "Polyline", "Point"])) {
            // Masks are not exported by roitool; anything else is unmapped.
            print("Skipping unsupported shape type: ${type}")
            continue
        }
        def get = { String suffix -> omexml."get${type}${suffix}"(roiIdx, s) }

        // Explicit null checks: Groovy treats 0 as falsy, so `?:` would turn channel 0 into -1.
        def cVal = get("TheC"), zVal = get("TheZ"), tVal = get("TheT")
        int c = cVal == null ? -1 : cVal.numberValue.intValue()
        int z = zVal == null ?  0 : zVal.numberValue.intValue()
        int t = tVal == null ?  0 : tVal.numberValue.intValue()
        def plane = c >= 0 ? ImagePlane.getPlaneWithChannel(c, z, t) : ImagePlane.getPlane(z, t)
        out.plane = plane
        // Shape-level affine: maps shape coordinates into image coordinates (OME spec).
        // PathViewer stores rotated shapes this way; ignoring it leaves them unrotated
        // and offset.
        def at = toJtsTransform(get("Transform"))
        int othersBefore = out.others.size()

        def strokeColor = get("StrokeColor")
        if (strokeColor != null && out.color == null)
            out.color = ColorTools.packARGB(strokeColor.alpha, strokeColor.red,
                                            strokeColor.green, strokeColor.blue)
        def locked = get("Locked")
        if (locked != null && out.locked == null) out.locked = locked
        // ADR-014 maps the object name onto shape Text. PathViewer-drawn ROIs carry
        // their label there too, and have no MapAnnotation at all.
        def text = get("Text")?.toString()?.trim()
        if (text && out.text == null) out.text = text

        switch (type) {
            case "Polygon":
                def coords = parsePoints(omexml.getPolygonPoints(roiIdx, s))
                if (coords.length < 3) break
                if (at) coords = coords.collect { at.transform(it, new Coordinate()) } as Coordinate[]
                // The push script marks interior rings with ".hole." in the Shape ID
                if ((omexml.getPolygonID(roiIdx, s) ?: "").contains(".hole.")) out.holes << coords
                else out.exteriors << coords
                break

            case "Rectangle":
                double x = omexml.getRectangleX(roiIdx, s), y = omexml.getRectangleY(roiIdx, s)
                out.others << [roi: ROIs.createRectangleROI(x, y,
                        omexml.getRectangleWidth(roiIdx, s), omexml.getRectangleHeight(roiIdx, s), plane),
                    color: out.color, locked: out.locked, text: text]
                break

            case "Ellipse":
                double rx = omexml.getEllipseRadiusX(roiIdx, s), ry = omexml.getEllipseRadiusY(roiIdx, s)
                out.others << [roi: ROIs.createEllipseROI(
                        omexml.getEllipseX(roiIdx, s) - rx, omexml.getEllipseY(roiIdx, s) - ry,
                        rx * 2, ry * 2, plane),
                    color: out.color, locked: out.locked, text: text]
                break

            case "Line":
                out.others << [roi: ROIs.createLineROI(
                        omexml.getLineX1(roiIdx, s), omexml.getLineY1(roiIdx, s),
                        omexml.getLineX2(roiIdx, s), omexml.getLineY2(roiIdx, s), plane),
                    color: out.color, locked: out.locked, text: text]
                break

            case "Polyline":
                def pts = parsePoints(omexml.getPolylinePoints(roiIdx, s))
                        .collect { new Point2(it.x, it.y) }
                if (pts.size() >= 2)
                    out.others << [roi: ROIs.createPolylineROI(pts, plane),
                        color: out.color, locked: out.locked, text: text]
                break

            case "Point":
                out.others << [roi: ROIs.createPointsROI(
                        omexml.getPointX(roiIdx, s), omexml.getPointY(roiIdx, s), plane),
                    color: out.color, locked: out.locked, text: text]
                break
        }
        // A transformed rectangle/ellipse is no longer axis-aligned, so it becomes a polygon
        if (at) for (int i = othersBefore; i < out.others.size(); i++)
            out.others[i].roi = GeometryTools.geometryToROI(at.transform(out.others[i].roi.getGeometry()), plane)
    }
    return out
}

/** OME AffineTransform → JTS, or null when absent. */
AffineTransformation toJtsTransform(t) {
    if (t == null) return null
    def m = [t.a00, t.a01, t.a02, t.a10, t.a11, t.a12]
    if (m.any { it == null }) return null
    m = m.collect { it as double }
    return new AffineTransformation(m[0], m[1], m[2], m[3], m[4], m[5])
}

/** "x,y x,y ..." → Coordinate[] */
Coordinate[] parsePoints(String points) {
    if (!points) return new Coordinate[0]
    return points.trim().split(/\s+/).findAll { it }.collect { p ->
        def xy = p.split(/,/)
        new Coordinate(Double.parseDouble(xy[0]), Double.parseDouble(xy[1]))
    } as Coordinate[]
}

/**
 * Builds one (multi)polygon from exterior rings plus any interior rings, assigning
 * each hole to the exterior that contains it. Rings are closed if the source did not.
 */
Geometry buildAreaGeometry(List exteriors, List holes, GeometryFactory gf) {
    def close = { Coordinate[] c ->
        if (c.length > 1 && !c[0].equals2D(c[c.length - 1])) {
            def list = new ArrayList<Coordinate>(Arrays.asList(c))
            list.add(new Coordinate(c[0]))
            return list as Coordinate[]
        }
        return c
    }
    def holeRings = holes.collect { gf.createLinearRing(close(it)) }
    def polys = exteriors.collect { ext ->
        def shell = gf.createLinearRing(close(ext))
        def shellPoly = gf.createPolygon(shell)
        def mine = holeRings.findAll { shellPoly.covers(gf.createPolygon(it)) }
        holeRings.removeAll(mine)
        mine ? gf.createPolygon(shell, mine as org.locationtech.jts.geom.LinearRing[])
             : shellPoly
    }
    def geom = polys.size() == 1 ? polys[0] : GeometryTools.union(polys)
    return geom.isValid() ? geom : geom.buffer(0)
}

/**
 * Inverts the Pageant Exclude convention (ADR-017): subtracts each Exclude geometry
 * that this geometry spatially covers. Association is spatial, exactly as Pageant
 * resolves it, so it is only reliable for non-overlapping annotations.
 */
Geometry subtractExcludes(Geometry geom, List excludes) {
    def result = geom
    excludes.each { ex ->
        if (result.covers(ex.getInteriorPoint())) result = result.difference(ex)
    }
    return result
}

/** Builds an annotation or detection from the restored qupath:* MapAnnotation pairs. */
PathObject makeObject(roi, Map props, String roiName, Integer color, Boolean locked,
                      String shapeText = null) {
    if (roi == null || roi.isEmpty()) return null

    def className = props["qupath:class"] ?: props["class"]
    if (className == null && roiName != null && !"Exclude".equalsIgnoreCase(roiName)
            && !GENERIC_ROI_NAMES.any { it.equalsIgnoreCase(roiName) })
        className = roiName
    def pathClass = className ? PathClass.fromString(className as String,
        color == null ? null : ColorTools.packRGB(ColorTools.red(color), ColorTools.green(color),
                                                  ColorTools.blue(color))) : null

    // Cells would need a second nucleus ROI; import them as plain detections.
    def obj = props["qupath:is-detection"] == "true" ?
        PathObjects.createDetectionObject(roi, pathClass) :
        PathObjects.createAnnotationObject(roi, pathClass)

    // Name: the qupath:name pair if the push script wrote one, else the shape's own
    // Text — that is where a PathViewer label such as "Hotspot1" lives.
    if (props["qupath:name"]) obj.setName(props["qupath:name"] as String)
    else if (shapeText) obj.setName(shapeText)
    if (pathClass == null && color != null) obj.setColor(color)
    if (locked != null) obj.setLocked(locked)
    props.each { k, v ->
        if (k.toString().startsWith("qupath:metadata:"))
            obj.getMetadata().put(k.toString().substring("qupath:metadata:".length()), v as String)
    }
    return obj
}
