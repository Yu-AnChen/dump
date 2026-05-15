# QuPath OME-XML / OMERO Integration — Decision Log

This document records design decisions made during development of the QuPath GeoJSON → OME-XML
tooling and the OMERO ROI upload pipeline, including rationale and rejected alternatives, so we
don't re-litigate solved problems.

---

## ADR-001 · Propagate `properties.name` to ROI `Name` and shape `Text`

**Status:** Refined — see ADR-014 for current mapping in `send_rois_to_omero.groovy`

**Context:**  
The original `OME_XML_export.groovy` (Glencoe Software, 2019) set `ROI/@Name` exclusively from
`path.pathClass.name`. QuPath GeoJSON exports carry the user-visible object name in
`properties.name` (e.g. `"1320554B_ROI_3 n=3"`), which was being silently dropped.

**Initial decision:**  
1. `omeROI.setName()` uses `path.getName()` first, falling back to `path.pathClass.name`.
2. `shape.setText(path.getName())` is called for every shape type.

This conflated name and class into a single field. The mapping was later separated so class and
name occupy distinct OME-XML attributes — see ADR-014.

**Rejected alternatives:**  
- Storing only in `MapAnnotation` `qupath:name` key — that was already working, but downstream
  tools that read native OME-XML attributes (not MapAnnotation pairs) would miss the name.

---

## ADR-002 · Headless batch processing via standalone Groovy script, not QuPath CLI

**Status:** Implemented as `geojson_to_ome_xml.groovy`

**Context:**  
The original script has two hard GUI dependencies that block headless execution:
- `QPEx.currentViewer.getOverlayOptions()` — requires an active QuPath viewer
- `Dialogs.promptToSaveFile(...)` — opens a file picker dialog

Three approaches were evaluated for batch GeoJSON → OME-XML conversion:

| Option | Approach | Verdict |
|--------|----------|---------|
| A | `QuPath script --image <img> myscript.groovy` | ❌ Requires an image to be open; awkward for GeoJSON-only workflows |
| B | `QuPath --headless script` with patched GUI calls | ⚠️ Still depends on QuPath runtime; harder to distribute |
| C | Pure Groovy script using OME jars from QuPath's `lib/app/` | ✅ Chosen |

**Decision:**  
Write `geojson_to_ome_xml.groovy` as a standalone script that:
- Reads GeoJSON directly (no QuPath objects, no GUI)
- Uses only OME Java libraries already bundled in QuPath (`lib/app/*.jar`)
- Accepts glob-pattern input and optional `--outdir` flag
- Runs via `groovy -cp "$QUPATH_LIB/*.jar" geojson_to_ome_xml.groovy *.geojson`

**Rationale:**  
The conversion is purely a data transformation (GeoJSON geometry + properties → OME-XML model).
QuPath's image-processing stack is not needed. Reusing the bundled jars avoids version mismatches
and extra downloads.

---

## ADR-003 · GeoJSON → OME-XML field mappings

**Status:** Implemented in `geojson_to_ome_xml.groovy`

**Context:**  
QuPath's GeoJSON export has several non-standard extensions. The mappings below were established
by reading the original script and inspecting real exported files.

| GeoJSON field | OME-XML destination | Notes |
|---|---|---|
| `properties.name` | shape `Text`; `ROI/@Name` fallback when no class | See ADR-014 |
| `properties.classification.name` | `ROI/@Name`; shape `Text` fallback when no name | See ADR-014 |
| `properties.classification.color` | `StrokeColor` | Packed ARGB integer → R,G,B,A |
| `properties.color` | `StrokeColor` | Per-object override; takes priority over class color |
| `properties.objectType` | `qupath:is-annotation` / `qupath:is-detection` MapPairs | Also drives stroke width default |
| `properties.measurements` | `qupath:metadata:*` MapPairs | Supports both `{key:val}` and `[{name,value}]` forms |
| `feature.id` (UUID) | `qupath:id` MapPair | |
| `geometry.isEllipse: true` on Polygon | OME `Ellipse` | Bounding box recovered from polygon ring min/max |
| `geometry.isRectangle: true` on Polygon | OME `Rectangle` | Bounding box recovered from polygon ring |
| `Polygon` with holes | Outer ring + each hole as separate `Polygon` shapes in same `Union`; hole shapes have `.hole.` in their Shape ID | OME has no native hole concept; Pageant variant splits into an "Exclude" ROI instead — see ADR-017 |
| `MultiPolygon` | One `Polygon` shape per part in same `Union` | |
| `LineString` (2 pts) | OME `Line` | |
| `LineString` (>2 pts) | OME `Polyline` | |
| `Point` / `MultiPoint` | OME `Point` per coordinate | |

**Known gaps / not yet mapped:**
- `GeometryROI` (wand tool output) → mask encoding; present in GUI script but not ported to
  headless script because it requires `BufferedImage` / AWT rendering, which needs display context.

---

## ADR-004 · No dependency on QuPath PathObject model in headless script

**Status:** Implemented

**Context:**  
The GUI script operates on QuPath's `PathROIObject` / `PathAnnotationObject` model, which is
populated by QuPath from GeoJSON on load. The headless script must parse GeoJSON directly.

**Decision:**  
Parse GeoJSON using Groovy's built-in `groovy.json.JsonSlurper`. No third-party JSON library
needed. All geometry and property extraction is done from the raw parsed map, not through QuPath
model objects.

**Implication:**  
The headless script is structurally different from the GUI script — they share the OME model
calls but diverge in how they obtain geometry and properties. This is intentional and not a bug.

---

## ADR-005 · Ellipse recovery from polygon ring

**Status:** Implemented

**Context:**  
QuPath exports ellipses as polygon approximations (96-point rings) with a `"isEllipse": true`
flag on the geometry object. OME-XML has a native `Ellipse` element. We want to round-trip
correctly.

**Decision:**  
Recover the bounding box from the polygon ring's coordinate min/max, then derive:
- `cx = (xmin + xmax) / 2`
- `cy = (ymin + ymax) / 2`
- `rx = (xmax - xmin) / 2`
- `ry = (ymax - ymin) / 2`

This matches what QuPath originally used to generate the approximated polygon.

**Rejected alternatives:**  
- Storing the polygon as-is — technically valid OME-XML, but loses the semantic "this is an
  ellipse" information and produces larger files.
- Fitting an ellipse to the ring points — unnecessarily complex; min/max is exact for
  axis-aligned ellipses, which is all QuPath produces.

---

*Last updated: 2026-05-13*

---

## ADR-006 · SSL certificate fix: patch QuPath OMERO extension, not system keystore

**Status:** Implemented (patched extension JAR)

**Context:**  
The target OMERO server uses a self-signed or privately-signed TLS certificate. The QuPath OMERO
extension's `HttpClient` threw a `PKIX path building failed` error on every connection attempt.

Three approaches were evaluated:

| Option | Approach | Verdict |
|--------|----------|---------|
| A | Import the server's `.pem` into QuPath's bundled `cacerts` via `keytool` | ⚠️ Permission denied on macOS sealed app bundle; breaks on QuPath updates |
| B | Set `JAVA_OPTS=-Djavax.net.ssl.trustStore=...` to a custom truststore | ⚠️ Fragile; hard to guarantee QuPath picks it up at launch |
| C | Patch `RequestSender.java` to inject a trust-all `SSLContext` | ✅ Chosen |

**Decision:**  
Patch two locations in the extension source to inject a `createTrustAllSSLContext()` helper:
1. The main `httpClient` field — add `.sslContext(createTrustAllSSLContext())`
2. The temporary `httpClient` inside `isLinkReachable()` — same addition

Rebuild with `./gradlew clean build`, drag the new JAR from `build/libs/` onto QuPath.

**Why trust-all instead of importing the cert:**  
The OMERO server is on a trusted internal network. Importing the cert was blocked by macOS app
bundle permissions. Trust-all is acceptable for this controlled environment and avoids re-doing
the fix on every QuPath update.

**Warning:** Do not use the trust-all build against public/untrusted OMERO servers.

---

## ADR-007 · ROI write endpoint: iViewer `persist_rois` is unavailable; OMERO JSON API also ruled out

**Status:** Dead end — both HTTP paths are blocked; see ADR-008 for resolution

**Context:**  
After the SSL fix, sending ROIs from QuPath via the OMERO extension failed with a 404. The
Glencoe fork of the extension hardcodes `/iviewer/persist_rois/` as its only ROI write endpoint.

Investigation found no viable HTTP/REST path for writing ROIs to a stock OMERO server:

| Endpoint | Status | Reason |
|---|---|---|
| `/iviewer/persist_rois/` | ❌ 404 | iViewer not installed on target server |
| `POST /api/v0/m/rois/` | ❌ Not supported | OMERO JSON API only supports creating Projects, Datasets, Screens — not ROIs |
| `/webgateway/...` | ❌ Read-only | No write endpoint exists |

**Installing iViewer** was ruled out — no admin access to the target server.

**Patching `IViewerApi.java`** to redirect to `/api/v0/m/rois/` was started but abandoned when
the OMERO docs confirmed that endpoint does not support ROI creation at all.

**Key lesson:** There is no web-only path for writing ROIs to OMERO. The ICE/Blitz protocol
(port 4064) is the only supported write path for ROIs.

---

## ADR-008 · ROI upload strategy: ome-omero-roitool via ICE/Blitz, integrated into QuPath script

**Status:** Implemented as `send_rois_to_omero.groovy`

**Context:**  
After exhausting HTTP/REST options (ADR-007), the ICE/Blitz API on port 4064 is the only
supported write path for ROIs. Glencoe's `ome-omero-roitool` wraps this and accepts OME-XML as
input — which we already produce.

**Decision:**  
`send_rois_to_omero.groovy` is a self-contained QuPath script that:
1. Auto-downloads and caches `ome-omero-roitool` v0.2.7 (ZIP from GitHub) on first run,
   stored in the QuPath user directory — no manual install needed
2. Extracts OMERO server host and image ID from the currently open image URI automatically
3. Passes `--key <session-key>` for auth — obtains an ICE session UUID via web login
   (`/api/v0/login/`) and caches it; see ADR-012 for the full auth flow
4. Exports all annotations + detections to a temp OME-XML (self-contained; no external script)
5. Calls `ome-omero-roitool import` as a subprocess
6. Cleans up the temp file on exit

**Why `--key` instead of `--user`/`--password`:**  
The web login endpoint returns an ICE session UUID that roitool accepts directly. This avoids
storing passwords on disk and avoids re-prompting on every run. See ADR-012.

**Why ICE port 4064 and not websockets:**  
The target server exposes port 4064. Websocket tunneling would require a different roitool
build configuration and was not needed.

**Rejected alternatives:**
- Direct ICE/Blitz from Groovy — requires adding OMERO client JARs to QuPath's classpath manually
- All web-based approaches — ruled out in ADR-007

---

*Last updated: 2026-05-13*

---

## ADR-009 · ICE shutdown hook kills QuPath: subprocess + auto-downloaded JRE (Adoptium)

**Status:** Implemented in `send_rois_to_omero.groovy`

**Context:**  
`ome-omero-roitool` uses the OMERO ICE/Blitz client, which registers a JVM shutdown hook that
calls `System.exit(0)` on clean finish. When invoked as a subprocess this is fine, but when
invoked in-process inside the running QuPath JVM it would kill QuPath entirely.

**First attempt: SecurityManager trap**  
Installed a custom `SecurityManager` that intercepts `checkExit()` and throws a
`SecurityException` instead, then restores the original manager in `finally`. This is the
classic pattern for embedding tools that call `System.exit()`.

**Why it failed:**  
`SecurityManager` was deprecated in Java 17 and **completely removed in Java 18+**. QuPath 0.7+
runs on a recent JVM, so `System.setSecurityManager()` throws `UnsupportedOperationException`
at runtime. The approach is dead on modern QuPath.

**Second attempt: subprocess with QuPath's bundled JRE**  
Run roitool as an external subprocess using the JRE bundled inside QuPath. Problem: QuPath
packages its runtime as a **jlink image**, which strips out the `bin/java` executable — only
`Contents/MacOS/java` (a launcher shim) exists on macOS (see ADR-010 for full resolution chain).
This attempt was abandoned.

**Third attempt: in-process via daemon thread + reflection**  
Load the roitool JARs into a child `URLClassLoader` and invoke `main()` via reflection on a
daemon thread, using a `CountDownLatch` for completion. ICE's `System.exit()` terminates the
thread rather than the JVM. This approach worked mechanically but required setting the thread
context classloader before invocation (ADR-011) and still surfaced classloader fragility
(ADR-013). It was ultimately abandoned in favour of the subprocess approach below.

**Final decision: subprocess with auto-downloaded JRE (Adoptium)**  
`send_rois_to_omero.groovy` runs roitool as a `ProcessBuilder` subprocess and supplies
a `JAVA_HOME` by checking for a previously cached Adoptium JRE in the QuPath user directory,
downloading one on first run if absent (~50 MB, one-time). See ADR-010 for the full resolution
history.

This eliminates the external-dependency objection to the subprocess approach — the JRE is
self-provisioned. `System.exit()` in the subprocess kills only the child process; QuPath
remains alive.

**What was tried and why each was rejected:**

| Approach | Verdict | Reason |
|---|---|---|
| `SecurityManager` trap | ❌ | Removed in Java 18+; not available in modern QuPath |
| Subprocess with QuPath JRE | ❌ | jlink image has no `bin/java`; needs system Java |
| In-process daemon thread + reflection | ❌ | Classloader fragility with ICE; see ADR-011, ADR-013 |
| Subprocess with system Java | ⚠️ | Works, but requires user to have Java installed |
| Subprocess with auto-downloaded Adoptium JRE | ✅ Chosen | Self-provisioned; exit() kills subprocess only |

---

## ADR-010 · JAVA_HOME resolution: auto-download JRE from Adoptium

**Status:** Resolved in `send_rois_to_omero.groovy` (supersedes earlier "skip subprocess" conclusion)

**Context:**  
During the subprocess approach (see ADR-009), we spent time trying to resolve a usable
`JAVA_HOME` to pass to the roitool shell script. The resolution chain attempted was:

1. `$JAVA_HOME` env var — not set when QuPath launches from the macOS dock/app bundle
2. `System.getProperty("java.home")` — returns QuPath's jlink runtime root; no `bin/java` there
3. `ProcessHandle.current().info().command()` — returns `Contents/MacOS/java` launcher shim;
   walking up the directory tree did not reliably yield a directory with a proper `bin/java`
4. `/usr/libexec/java_home` (macOS utility) — works if a system JDK is installed, but
   introduces the system Java dependency we were trying to avoid
5. `which java` — same problem

**Key finding:**  
QuPath's jlink runtime intentionally omits a standalone `java` binary. There is no reliable,
cross-platform, zero-external-dependency way to obtain a `java` executable from within a
QuPath script when QuPath was launched as a `.app` bundle on macOS.

**Initial (wrong) conclusion:**  
Abandon the subprocess approach in favour of in-process classloader invocation (ADR-009 third
attempt). That path turned out to be fragile for other reasons (ADR-011, ADR-013).

**Final decision:**  
Resolve `JAVA_HOME` by auto-downloading a JRE from the Adoptium API on first run, cached in
the QuPath user directory. Resolution in `send_rois_to_omero.groovy`:

1. Previously cached Adoptium JRE in `<userDir>/jre-<version>-<os>-<arch>/`
2. Download from `api.adoptium.net/v3/binary/latest/...` and cache

macOS Adoptium JREs are `.app` bundles; `java` is at `Contents/Home/bin/java`. Linux/Windows
have `bin/java` at the archive root after stripping the top-level directory entry.

**Implication:**  
The subprocess approach is viable as long as a `java` binary can be located or provisioned.
Auto-download is the zero-configuration, self-contained path.

---

*Last updated: 2026-05-13*

---

## ADR-011 · ICE SSL plugin classloader conflict (in-process path — superseded)

**Status:** Superseded — in-process reflection approach was abandoned; see ADR-009

**Context:**  
During the in-process reflection attempt (ADR-009 third attempt), a new error appeared:
`IceSSL.PluginFactory` could not find its own classes at runtime. ICE's SSL plugin uses
`Class.forName()` and `ServiceLoader` internally for plugin discovery, both of which resolve
against the **thread context classloader** — not the caller's classloader. Since the roitool
classes were loaded into a child `URLClassLoader` but the thread's context classloader was still
QuPath's classloader, ICE's plugin factory couldn't see its own classes across the boundary.

**Fix applied at the time:**  
Before invoking `mainMethod.invoke()`, set the thread's context classloader to the roitool
classloader, and restore the original in `finally`:

```groovy
def originalContextLoader = Thread.currentThread().contextClassLoader
Thread.currentThread().contextClassLoader = roitoolClassLoader
try {
    mainMethod.invoke(null, [args] as Object[])
} finally {
    Thread.currentThread().contextClassLoader = originalContextLoader
}
```

**Why this ADR is retained:**  
The general principle is still valid: any library loaded via a child `URLClassLoader` and
invoked by reflection requires the thread context classloader to be set before the call.
This applies to JNDI, JDBC, ICE, and any other `ServiceLoader`-based plugin architecture.

**Why the approach was abandoned:**  
The classloader fix resolved the SSL plugin error, but the in-process path continued to surface
fragility (ICE's `System.exit()` behaviour, stdout stream redirection). The subprocess +
Adoptium JRE approach (ADR-009, ADR-010) is cleaner and is the current implementation.

---

## ADR-012 · OMERO auth strategy: web login to /api/v0/login/ exchanges credentials for ICE session UUID; cache UUID, never password

**Status:** Implemented

**Context:**  
The roitool `--key` flag requires an ICE session UUID. We needed a way to obtain and cache
this without storing the user's password on disk between runs.

**Key discovery:**  
OMERO's JSON API `/api/v0/login/` endpoint (POST with username + password + CSRF token) returns
a `sessionUuid` in the response JSON. This UUID **is** the ICE session key — the same token
used by `omero.client.joinSession()`. This means a single web login call gives us the ICE key
without needing the ICE/Blitz protocol at all for auth.

**Auth flow implemented:**
1. On first run: prompt username + password via QuPath dialog
2. POST to `/api/v0/token/` to get CSRF token (required by Django)
3. POST to `/api/v0/login/` with credentials + CSRF token
4. Parse `sessionUuid` from response → this is the ICE key
5. Cache `{username, session_key}` to `~/.omero_qupath_creds` (Java `.properties` file)
6. On subsequent runs: load key from cache, skip prompt entirely
7. On session expiry: roitool exits with permission error → cache cleared → next run re-prompts

**What is and isn't stored on disk:**
- ✅ Cached: OMERO username, ICE session UUID
- ❌ Never stored: password

**Bug found and fixed during development:**  
The initial implementation had two bugs:
- `"Using cached session key"` was printed twice (duplicate print statement)
- The session key was never written to cache after a successful web login — only the failed
  path wrote to disk. Fixed by writing the cache immediately after `sessionUuid` is parsed.

---

## ADR-013 · QuPath crash after roitool completes: caused by System.exit() from ICE shutdown hook

**Status:** Resolved by subprocess isolation (ADR-009 final decision)

**Context:**  
After getting the first successful ROI upload (confirmed in OMERO), QuPath crashed/quit at the
end of the run. The crash was caused by ICE's JVM shutdown hook calling `System.exit(0)` after
a clean finish, which terminated the entire QuPath JVM.

**Initial suspicion (wrong):**  
`System.setOut`/`System.setErr` stream redirection was suspected — QuPath's JavaFX UI thread
has strict expectations about stdout/stderr, and stream manipulation during ICE teardown could
destabilize the JVM.

**Actual cause:**  
ICE's shutdown hook calls `System.exit(0)`. When roitool runs in-process (via reflection on the
QuPath JVM), this exits QuPath itself, not just the roitool invocation.

**Intermediate fix (abandoned):**  
Running roitool on a daemon thread with a `CountDownLatch` was tried; `System.exit()` would
terminate the thread rather than the JVM. This was fragile alongside the classloader issues
(ADR-011) and was abandoned.

**Final resolution:**  
Run roitool as a `ProcessBuilder` subprocess with a `JAVA_HOME` supplied from the auto-downloaded
Adoptium JRE (ADR-009, ADR-010). `System.exit()` in the subprocess kills only the child
process — QuPath is completely unaffected. Verified by `test_shutdown_hook3.groovy`, which
prints `"Done — if you see this, QuPath survived!"` after the subprocess exits.

**Significance:**  
This was the final blocker. Once the subprocess + Adoptium JRE approach was in place, uploads
completed successfully and QuPath remained alive.

---

## ADR-014 · Name/class mapping: ROI.name ← class, Shape.text ← name

**Status:** Implemented in `send_rois_to_omero.groovy`

**Context:**  
ADR-001's initial mapping collapsed name and class onto the same OME-XML field (`ROI/@Name` and
`shape/Text` both set to `path.getName()`). This means classification information was only
accessible via the MapAnnotation `qupath:class` key, not via native OME-XML attributes.

**Decision:**  
When both name and class are present, use each for the attribute it maps to most naturally:
- `ROI/@Name` ← `path.pathClass.name` (the classification label — shared across object type)
- shape `Text` ← `path.getName()` (the user-assigned name — unique per object)

When only one is present, use it for both attributes.

**Rationale:**  
`ROI/@Name` is a category label in most viewers (groups ROIs by type). `shape/Text` is a
per-shape annotation. Mapping class → ROI name and object name → shape text aligns with the
semantic intent of both fields and matches how downstream tools (including Pageant) interpret them.

**Rejected alternative:**  
Keeping ADR-001 mapping (name takes priority for both) — loses class information from native
OME attributes when an object has both name and class.

---

## ADR-015 · Split web/ICE hostname: OMERO_ICE_HOST config + connection-refused retry

**Status:** Implemented in `send_rois_to_omero.groovy`

**Context:**  
Some OMERO deployments expose the web frontend and ICE/Blitz service on different hostnames
(e.g. HMS: web = `omero.hms.harvard.edu`, ICE = `omero-app.hms.harvard.edu:4064`). The image
URL only contains the web hostname, but roitool needs the ICE hostname.

**Decision:**  
1. `OMERO_ICE_HOST` config constant at the top of the script — set once for a given deployment.
2. If unset, default to the hostname parsed from the image URL.
3. If the connection is refused, prompt the user for the correct ICE hostname and retry once.
4. On success, cache the effective ICE hostname alongside the session key so future runs use it
   automatically without prompting.

**Rejected alternative:**  
Requiring the user to always set `OMERO_ICE_HOST` explicitly — too much friction for users who
discover the mismatch only when the first run fails.

---

## ADR-016 · Subprocess hang fix: background reader thread + fatalSeen early kill

**Status:** Implemented in `send_rois_to_omero.groovy`

**Context:**  
When roitool fails with a fatal ICE error (e.g. `ConnectionRefusedException`), its daemon
threads keep the subprocess alive indefinitely — the process never exits on its own. The script
was hanging for the full 120-second timeout on every failed attempt.

**Decision:**  
- Stream subprocess output on a background reader thread (necessary anyway to prevent
  pipe-buffer deadlock on long output).
- Track an `AtomicBoolean fatalSeen` set when fatal error strings appear in the output
  (`ConnectionRefusedException`, `PermissionDeniedException`, `Connection refused`).
- In the polling loop, force-kill the subprocess 5 seconds after `fatalSeen` is true,
  rather than waiting for the full 120-second timeout.

**Result:**  
Failed connections are detected and killed within ~5 seconds instead of ~120 seconds.

---

## ADR-017 · Pageant variant: "Exclude" ROI for polygon holes

**Status:** Implemented as `send_rois_to_omero_pageant.groovy`

**Context:**  
Pageant (Glencoe Software) computes per-object measurements on OMERO ROIs. It does not natively
handle polygon holes — interior ring shapes rendered as sibling shapes in the same Union (the
base script's approach) are measured as filled regions, not subtracted.

Pageant's workaround: a ROI with `ROI/@Name = "Exclude"` is spatially subtracted from any
enclosing exterior ROI when computing measurements.

**Decision:**  
Create a separate script `send_rois_to_omero_pageant.groovy` that differs only in the
`AreaROI`/`GeometryROI` handling:
- Exterior rings → main ROI (same as base script, minus hole shapes)
- All interior rings (holes) → separate ROI with `ROI/@Name = "Exclude"`, same MapAnnotation
  metadata as the parent object, and Shape IDs preserving the `.hole.` naming convention

`addROI` returns `int` (ROIs emitted) instead of `void`, so the index counter accumulates
correctly when a single QuPath object produces two ROIs.

**Why no explicit exterior–interior link:**  
Pageant resolves the association spatially — an Exclude ROI is subtracted from whichever
exterior ROI spatially contains it. This is sufficient for non-overlapping annotations (the
common case in pathology).

**Rejected alternative:**  
A runtime flag in the base script — kept as a separate file to avoid conditional complexity
in the hot path and to make each script independently readable.

---

## ADR-018 · Windows roitool execution: bypass .bat with java.exe + @argfile

**Status:** Implemented in `send_rois_to_omero.groovy` and `send_rois_to_omero_pageant.groovy`

**Context:**  
On Windows, roitool is distributed as a `.bat` file. When the script first called the `.bat`
via `ProcessBuilder`, roitool exited with code 255 and printed `The input line is too long`.
The root cause: `cmd.exe` has an **8191-character command-line limit**. The `.bat` expands its
entire classpath (dozens of JARs) into a single `java -classpath <...>` invocation, which
routinely exceeds this limit.

macOS and Linux are unaffected — bash has an `ARG_MAX` of ~2 MB. On those platforms the script
continues to invoke the shell script directly via `/bin/bash`.

**Debugging iterations on Windows:**

1. **Exit code 255 / "input line too long"**  
   Calling the `.bat` directly via `ProcessBuilder`. The `.bat`'s `set CLASSPATH=...` line,
   when substituted by `cmd.exe`, exceeded 8191 chars.  
   Fix direction: bypass the `.bat` and invoke `java.exe` directly.

2. **`arraycopy: element type mismatch` (GString type error)**  
   After switching to `["${javaHome}\\bin\\java.exe", ...]`, the list contained Groovy `GString`
   instances rather than `java.lang.String`. `ProcessBuilder` requires a `List<String>` and the
   array copy failed at native dispatch.  
   Fix: concatenate strings explicitly — `javaHome.toString() + "\\bin\\java.exe"` — so the
   list contains real `String` elements.

3. **`ClassNotFoundException: com.glencoesoftware.roitool.Main`**  
   An `eachFile` scan of the `lib/` directory returned an empty list (silent iteration failure).  
   Fix: parse the classpath directly from the `.bat` — find the `set CLASSPATH=` line, strip the
   prefix, replace `%APP_HOME%` with the real installation path. `eachFileRecurse` on `lib/`
   remains as a fallback if bat parsing finds nothing.

**Final solution:**  
Java supports `@argfile` — a file containing JVM flags, one per line, passed as `@path/to/file`.
This bypasses `cmd.exe`'s length limit entirely because the JVM launcher reads the file before
`cmd.exe` sees any classpath content.

```groovy
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
    def jars = []
    new File(appHome, "lib").eachFileRecurse { f ->
        if (!f.directory && f.name.endsWith(".jar")) jars << f.absolutePath
    }
    classpath = jars.join(';')
}
// Java argfile: backslashes are escape chars inside quotes → use forward slashes
cpArgFile = File.createTempFile("roitool_cp_", ".args")
cpArgFile.deleteOnExit()
cpArgFile.text = "-cp\n\"${classpath.replace('\\', '/')}\"\n"

cmd = [javaHome.toString() + "\\bin\\java.exe",
       "@" + cpArgFile.absolutePath,
       mainClass] + importArgs
```

Key details:
- `@argfile` is parsed by the JVM launcher, not `cmd.exe` — no length limit.
- Backslashes are escape characters inside quotes in Java argfiles; replacing with forward slashes
  is necessary and valid on Windows Java.
- The argfile is created in a temp location and deleted in `finally` after the subprocess exits.
- `mainClass` is parsed from the `.bat`'s invocation line; hardcoded fallback to
  `com.glencoesoftware.roitool.Main` if the regex doesn't match.

---

*Last updated: 2026-05-13*
