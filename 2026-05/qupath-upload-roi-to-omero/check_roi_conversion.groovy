/**
 * check_roi_conversion.groovy — does each GeoJSON match the OME-XML it came from? (ADR-025)
 *
 *   QuPath script check_roi_conversion.groovy -a <folder | file.ome.xml> [...]
 *
 * For every <name>.ome.xml with a sibling <name>.geojson (pull_rois_from_omero.groovy
 * output), two checks; any failure makes QuPath exit 1:
 *
 * 1. Coverage — every Shape attribute and child element must be listed below as
 *    either applied to geometry or deliberately ignored. A shape Transform went
 *    unnoticed because neither the converter nor any test knew about it; anything
 *    new now fails here until someone decides what it means.
 * 2. Geometry — each shape is rebuilt from the raw XML per the OME spec, independently
 *    of the converter (no OMEXMLMetadata), grouped as ADR-020 describes, and matched
 *    to a GeoJSON feature by Hausdorff distance.
 */
import groovy.xml.XmlSlurper
import org.locationtech.jts.algorithm.distance.DiscreteHausdorffDistance
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.Geometry
import org.locationtech.jts.geom.util.AffineTransformation
import org.locationtech.jts.util.GeometricShapeFactory
import qupath.lib.io.PathIO
import qupath.lib.roi.GeometryTools

// Attributes that define geometry, per shape type. Everything here is applied below.
GEOMETRY = [
    Rectangle: ["X", "Y", "Width", "Height"],
    Ellipse:   ["X", "Y", "RadiusX", "RadiusY"],
    Polygon:   ["Points"],
    Polyline:  ["Points"],
    Line:      ["X1", "Y1", "X2", "Y2"],
    Point:     ["X", "Y"],
]
APPLIED_ELEMENTS = ["Transform"]
// Present in real exports and deliberately not geometry. Add to these only after
// deciding the new item really does not move or reshape anything.
IGNORED_ATTRIBUTES = ["ID", "Text", "TheZ", "TheT", "TheC", "Locked", "FillColor", "FillRule",
                      "StrokeColor", "StrokeWidth", "StrokeWidthUnit", "StrokeDashArray",
                      "FontFamily", "FontSize", "FontSizeUnit", "FontStyle", "MarkerStart", "MarkerEnd"]
IGNORED_ELEMENTS = ["AnnotationRef"]
// Allowed outline deviation: ellipses are polygonised differently here and in QuPath.
TOLERANCE_PX = 1.0

def inputs = (binding.hasVariable('args') ? args as List : []).collect { new File(it as String) }
if (!inputs) throw new IllegalArgumentException("usage: -a <folder | file.ome.xml> [...]")
def xmls = inputs.collectMany { f ->
    f.directory ? (f.listFiles().findAll { it.name.endsWith(".ome.xml") }.sort { it.name }) : [f]
}
int failed = 0
xmls.each { xml ->
    def geojson = new File(xml.parentFile, xml.name.replaceAll(/\.ome\.xml$/, ".geojson"))
    if (!geojson.exists()) { println "SKIP  ${xml.name}: no ${geojson.name}"; return }
    def problems = checkPair(xml, geojson)
    if (problems.ok) println "OK    ${xml.name}: ${problems.summary}"
    else { failed++; println "FAIL  ${xml.name}: ${problems.summary}"; problems.details.each { println "        ${it}" } }
}
println "${xmls.size() - failed}/${xmls.size()} pair(s) passed"
if (failed) throw new IllegalStateException("${failed} OME-XML/GeoJSON pair(s) failed")

Map checkPair(File xmlFile, File geojsonFile) {
    def details = []
    def gf = GeometryTools.getDefaultFactory()
    def ome = new XmlSlurper(false, false).parse(xmlFile)

    // ── Expected objects, grouped as the converter is documented to (ADR-020) ──
    def expected = []      // [label, geometry]
    def merged = []        // per-ROI polygon unions, which Exclude ROIs subtract from
    def excludes = []
    ome.ROI.each { roi ->
        def isExclude = roi.@Name.text().equalsIgnoreCase("Exclude")
        def exteriors = [], holes = []
        roi.Union.children().each { s ->
            def type = s.name()
            if (!GEOMETRY.containsKey(type)) { details << "ROI '${roi.@Name}': unsupported shape ${type}"; return }
            s.attributes().keySet().findAll { !(it in GEOMETRY[type]) && !(it in IGNORED_ATTRIBUTES) }
                .each { details << "unhandled attribute ${type}@${it}" }
            s.children()*.name().findAll { !(it in APPLIED_ELEMENTS) && !(it in IGNORED_ELEMENTS) }
                .each { details << "unhandled element ${type}/${it}" }
            def g = transformed(shapeGeometry(s, type, gf), s.Transform)
            if (type == "Polygon") (s.@ID.text().contains(".hole.") && !isExclude ? holes : exteriors) << g
            else if (!isExclude) expected << ["${roi.@Name} ${type}", g]
        }
        if (!exteriors) return
        def area = gf.buildGeometry(exteriors).union()
        holes.each { area = area.difference(it) }
        if (isExclude) excludes << area else merged << ["${roi.@Name} Polygon", area]
    }
    merged.each { m ->
        excludes.each { ex -> if (m[1].covers(ex.getInteriorPoint())) m[1] = m[1].difference(ex) }
        expected << m
    }

    // ── Match against the GeoJSON features ──
    def actual = PathIO.readObjects(geojsonFile.toPath()).collect { it.getROI().getGeometry() }
    if (actual.size() != expected.size())
        details << "expected ${expected.size()} object(s), GeoJSON has ${actual.size()}"
    double worst = 0
    expected.each { label, Geometry g ->
        if (!actual) return
        def best = actual.min { DiscreteHausdorffDistance.distance(g, it, 0.05) }
        double d = DiscreteHausdorffDistance.distance(g, best, 0.05)
        worst = Math.max(worst, d)
        if (d > TOLERANCE_PX) details << "${label}: nearest GeoJSON feature is ${String.format('%.1f', d)} px off"
        actual.remove(best)
    }
    return [ok: details.isEmpty(), details: details.unique(),
            summary: "${expected.size()} object(s), max deviation ${String.format('%.2f', worst)} px" +
                     (details ? ", ${details.size()} problem(s)" : "")]
}

Geometry shapeGeometry(s, String type, gf) {
    def n = { String a -> s."@${a}".text() as double }
    def pts = { (s.@Points.text().trim().split(/\s+/)).collect { def xy = it.split(','); new Coordinate(xy[0] as double, xy[1] as double) } }
    switch (type) {
        case "Rectangle":
            def (x, y, w, h) = [n("X"), n("Y"), n("Width"), n("Height")]
            return gf.createPolygon([new Coordinate(x, y), new Coordinate(x + w, y), new Coordinate(x + w, y + h),
                                     new Coordinate(x, y + h), new Coordinate(x, y)] as Coordinate[])
        case "Ellipse":
            def f = new GeometricShapeFactory(gf)
            f.setCentre(new Coordinate(n("X"), n("Y")))
            f.setWidth(2 * n("RadiusX")); f.setHeight(2 * n("RadiusY")); f.setNumPoints(256)
            return f.createEllipse()
        case "Polygon":
            def c = pts()
            if (!c.first().equals2D(c.last())) c << new Coordinate(c.first())
            return gf.createPolygon(c as Coordinate[])
        case "Polyline": return gf.createLineString(pts() as Coordinate[])
        case "Line":     return gf.createLineString([new Coordinate(n("X1"), n("Y1")), new Coordinate(n("X2"), n("Y2"))] as Coordinate[])
        case "Point":    return gf.createPoint(new Coordinate(n("X"), n("Y")))
    }
}

// OME spec: x' = A00*x + A01*y + A02, y' = A10*x + A11*y + A12
Geometry transformed(Geometry g, t) {
    if (t.size() == 0) return g
    def a = { String k -> t."@${k}".text() as double }
    return new AffineTransformation(a("A00"), a("A01"), a("A02"), a("A10"), a("A11"), a("A12")).transform(g)
}
