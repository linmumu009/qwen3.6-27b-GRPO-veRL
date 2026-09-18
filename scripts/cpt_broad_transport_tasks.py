"""Historical statistical definitions applied only to explicit hypothetical records."""

def add_transport(add):
    u='road_capacity'
    prefix='Use the 2019 Eurostat/UNECE/ITF statistical definitions. '
    add(u,'train','mass_volume_binding',prefix+
        'A hypothetical goods vehicle has declared load capacity 8 tonnes and load volume 20 cubic metres. Shipment A weighs 7 tonnes and occupies 22 cubic metres. Both stated limits must be satisfied; no other limit is relevant.',[
        ('A satisfies the mass limit but exceeds the volume limit.',True,'Seven is within eight; 22 exceeds 20.'),
        ('A is feasible solely because its mass is below eight tonnes.',False,'It violates the separate volume limit.'),
        ('Twenty cubic metres is a goods mass limit.',False,'It is a volume, not mass.'),
        ('Load capacity and load volume are interchangeable quantities.',False,'The definitions use different physical dimensions.')])
    add(u,'train','gross_vs_goods',prefix+
        'A hypothetical vehicle has maximum permissible gross weight 18 tonnes. Its road-ready non-goods mass, including all required occupants and equipment, is 7 tonnes. Declared goods load capacity is 10 tonnes. Ignore volume and all other constraints.',[
        ('The gross-weight headroom alone is 11 tonnes.',True,'18 minus seven leaves eleven.'),
        ('The permitted goods mass under both supplied limits is 11 tonnes.',False,'The separate declared goods capacity limits it to ten.'),
        ('A ten-tonne goods load meets both stated mass limits.',True,'It equals goods capacity and makes gross mass 17.'),
        ('Gross vehicle weight excludes all occupants by definition.',False,'The cited definition includes the driver and permitted persons.')], [('18-7','11'),('7+10','17')])
    add(u,'train','road_train_definition',prefix+
        'A road train consists of a lorry and a trailer whose declared goods load capacities are 6 and 4 tonnes. Use the cited statistical load-capacity definition. Do not infer route permission or other operating limits not given.',[
        ('The defined road-train load capacity is ten tonnes.',True,'The cited definition sums the lorry and trailer capacities.'),
        ('The statistical sum itself establishes permission on every road.',False,'Route permissions are not specified by this definition.'),
        ('The capacities concern goods mass.',True,'Load capacity is maximum declared permissible goods weight.'),
        ('Component load volumes would require separate volume information.',True,'Mass capacities do not determine cubic volume.')], [('6+4','10')])
    add(u,'train','four_distinct_measurements',prefix+
        'A record separately reports permissible goods mass, available cubic space, maximum permissible loaded road-ready gross weight, and actual goods mass for one trip. No numerical values or equality between those entries is asserted.',[
        ('Permissible goods mass corresponds to load capacity.',True,'That is the cited definition.'),
        ('Available cubic space corresponds to load volume.',True,'Load volume concerns space available for goods.'),
        ('Maximum loaded road-ready weight includes vehicle and load.',True,'Gross weight is their total, with the cited occupant inclusions.'),
        ('Actual goods mass need not equal the maximum permissible goods mass.',True,'An actual load and a capacity are different measurements.')])
    add(u,'dev','integer_parcel_capacity',prefix+
        'A hypothetical vehicle permits six tonnes of goods and 10 cubic metres. Identical indivisible packages each weigh two tonnes and occupy three cubic metres. Every package must be carried whole, and both limits apply with no other restriction.',[
        ('At most three packages can be carried.',True,'Three use six tonnes and nine cubic metres; four exceed both limits.'),
        ('Three packages leave one cubic metre unused.',True,'Ten minus nine equals one.'),
        ('The spare cubic metre does not permit an additional whole package.',True,'A package needs three cubic metres and mass capacity is already full.'),
        ('Dividing cubic metres by tonnes gives the number of packages without package data.',False,'The unit conversion is not a package count.')], [('3*2','6'),('3*3','9'),('10-9','1')])
    add(u,'dev','volume_only_change',prefix+
        'Before a change, a vehicle has goods capacity eight tonnes and volume 12 cubic metres. A redesign raises only its available volume to 18 cubic metres; the goods mass limit stays eight tonnes. A shipment weighs nine tonnes and occupies 15 cubic metres.',[
        ('The redesign removes the volume violation but leaves the mass violation.',True,'Fifteen now fits in 18, but nine still exceeds eight.'),
        ('The declared mass limit remains eight tonnes.',True,'The scenario explicitly keeps it unchanged.'),
        ('The shipment remains infeasible under the two simultaneous limits.',True,'The mass limit is still exceeded.'),
        ('The original vehicle failed the volume requirement.',True,'Fifteen exceeded the original twelve.')])

    u='rail_track_line'
    prefix='Use only the 2019 Eurostat/UNECE/ITF railway statistical scope. '
    add(u,'train','line_and_parallel_tracks',prefix+
        'A maintained railway route is ten kilometres long and has two full-length parallel running tracks. There are no sidings, excluded stretches or gaps. Distinguish route-line length from cumulative running-track length.',[
        ('The line length is ten kilometres and the two tracks total twenty kilometres.',True,'Parallel tracks increase track length without doubling the route length.'),
        ('The line length is necessarily twenty kilometres because there are two tracks.',False,'Line and track measures differ.'),
        ('Only one of the parallel tracks can belong to a railway line.',False,'A line may contain one or more tracks.'),
        ('Track and line length are identical by definition in every network.',False,'Multiple tracks can compose one line.')], [('10*2','20')])
    add(u,'train','siding_scope',prefix+
        'A public railway has 20 kilometres of running track, a three-kilometre publicly accessible siding managed by the infrastructure manager, and a two-kilometre privately operated siding. No other tracks exist. Apply the cited track-length treatment of sidings.',[
        ('The public manager-operated siding contributes three kilometres to the specified track total.',True,'It satisfies the inclusion conditions.'),
        ('The privately operated siding must be included merely because it connects to the public network.',False,'Private sidings are excluded from the cited total.'),
        ('The specified track total is 23 kilometres.',True,'It includes the running tracks and eligible public siding only.'),
        ('A siding ceases to be private solely because it connects an industrial loading site to the public network.',False,'That connection is compatible with the private-siding definition.')], [('20+3','23')])
    add(u,'train','excluded_links',prefix+
        'An itinerary contains 30 kilometres of qualifying railway line, followed by eight kilometres on a wagon-carrying ferry, then 12 more kilometres of qualifying railway line. Count line length for these itinerary segments; no segment overlaps.',[
        ('The qualifying railway-line segments total 42 kilometres.',True,'Thirty plus twelve are railway line.'),
        ('The ferry segment becomes railway line because wagons are carried on it.',False,'Water stretches remain excluded even when they convey rolling stock.'),
        ('The full itinerary distance can exceed counted railway-line distance.',True,'It contains an excluded water leg.'),
        ('The source expressly distinguishes rail infrastructure from water carriage of wagons.',True,'Its railway-line exclusion addresses such ferries.')], [('30+12','42'),('30+8+12','50')])
    add(u,'train','classification_records',prefix+
        'Records describe an end-to-end running track between timetable stations, a line composed of two tracks, a branch siding, and a privately operated connection from an industrial loading facility to the public network. Evaluate these source-based classifications.',[
        ('The timetable-continuity track is consistent with a main/running track.',True,'It matches the source function between designated endpoints.'),
        ('A line can contain two tracks.',True,'A railway line comprises one or more tracks.'),
        ('A siding can branch off a running main track.',True,'That is the cited siding definition.'),
        ('The privately operated industrial connection is consistent with a private siding.',True,'It connects loading facilities to the public network under private operation.')])
    add(u,'dev','partial_double_tracking',prefix+
        'A railway line is 18 kilometres long. Its first six kilometres have two parallel running tracks and its remaining twelve have one. A separate two-kilometre siding is publicly accessible and managed by the infrastructure manager. There are no other or excluded track segments.',[
        ('Cumulative running-track length is 24 kilometres.',True,'Six times two plus twelve equals 24.'),
        ('The line length becomes 24 kilometres when the second track is added.',False,'The line route remains eighteen kilometres.'),
        ('Including the eligible siding gives 26 kilometres of track.',True,'The two-kilometre siding is added to 24.'),
        ('Including the separate siding in track length does not change the specified eighteen-kilometre line route.',True,'Track inclusion does not redefine the stated line route.')], [('6*2+12','24'),('24+2','26')])
    add(u,'dev','urban_scope_counterexample',prefix+
        'An inventory lists a metro-only urban line and a publicly accessible conventional railway running track. A compiler includes both in the cited A.I-01 railway-track total solely because both use rails. No cross-use or alternative scope is stated.',[
        ('The metro-only urban line is excluded under the cited track scope.',True,'A.I-01 excludes metro, tram and light-rail urban lines.'),
        ('Using rails alone is insufficient for inclusion under the cited scope.',True,'The definition includes explicit exclusions.'),
        ('The metro exclusion alone does not exclude the conventional running track.',True,'Each entry must be assessed against its own scope conditions.'),
        ('The conclusion remains limited to the named historical statistical scope.',True,'The definition is not a claim about every current local reporting rule.')])
