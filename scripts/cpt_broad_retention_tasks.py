"""Sixteen evaluation-only constructions in the frozen historical dev source pool."""

def add_retention(add):
    def keep(unit,design,q,options,checks=()):add(unit,'retention',design,q,options,checks)
    keep('retain_pest','external_change_scan',
        'A firm observes a national population shift, changes in external communications technology, its own overtime roster and its internal bin-label design. For a PEST macro-environment scan, distinguish the external observations from internal operating choices.',[
        ('The population shift is within the social macro-environment.',True,'PEST includes external social factors.'),
        ('External technology developments are within the technological macro-environment.',True,'That is one of the four PEST dimensions.'),
        ('The firm\'s own overtime roster alone is an external macro-environment factor.',False,'It is an internal operating choice.'),
        ('An internal bin-label design alone establishes a national political change.',False,'The facts do not support that external classification.')])
    keep('retain_safety','risk_not_guarantee',
        'A manager adds extra stock because deliveries sometimes fall short and demand forecasts are uncertain. The manager claims this guarantees that no conceivable future stockout can occur, regardless of disruption size.',[
        ('The extra stock serves the cited safety-stock purpose.',True,'It aims to mitigate stockout risk from supply or forecast uncertainty.'),
        ('The stated definition proves the unlimited no-stockout guarantee.',False,'Mitigating a risk does not establish protection against every disruption.'),
        ('Forecast uncertainty can be relevant to the reason for extra stock.',True,'The definition explicitly mentions forecasting supply and demand.'),
        ('Safety stock is defined as stock held only when all supply is perfectly known.',False,'The cited role includes uncertainty and shortfalls.')])
    keep('retain_abc','control_swap_audit',
        'Class assignments A, B and C are fixed. A control schedule gives A minimal records, B moderate records, and C very tight controls with accurate records. Compare the schedule with the archived ABC descriptions, without changing the assignments.',[
        ('A\'s minimal-record policy conflicts with the stated A description.',True,'A calls for very tight control and accurate records.'),
        ('B\'s moderate records are consistent with the stated B description.',True,'The source gives B moderate records.'),
        ('The source assigns the tightest control to C merely because C is third.',False,'Its C description uses the simplest controls and minimal records.'),
        ('The descriptions require identical record intensity for A and C.',False,'The classes expressly differ.')])
    keep('retain_uld','container_only_catalog',
        'A catalog uses the archived IATA ULD scope but lists only aircraft containers. A shipment is secured on an aircraft pallet with its net. No other packaging or aircraft equipment is under discussion.',[
        ('The pallet-and-net combination is within the cited ULD scope.',True,'The definition includes an aircraft pallet combined with its net.'),
        ('A container-only catalog omits an allowed ULD form.',True,'ULD is broader than aircraft container alone.'),
        ('The source establishes that any loose road pallet without a net is an aircraft ULD.',False,'It gives the specific aircraft pallet-and-net combination.'),
        ('Container and ULD must be synonymous in the cited scope.',False,'One is only one included form.')])
    keep('retain_epal','dimension_conversion',
        'A record explicitly identifies an EPAL 1 pallet. A form records footprint dimensions in centimetres and height in millimetres. Use the archived EPAL 1 dimensions; do not generalize to other pallet types.',[
        ('The footprint can be entered as 80 by 120 centimetres.',True,'800 by 1200 millimetres converts to 80 by 120 centimetres.'),
        ('The height entry is 144 millimetres.',True,'That is the stated EPAL 1 height.'),
        ('The given dimensions define every logistics pallet.',False,'They identify this type only.'),
        ('The footprint must be entered as 800 by 1200 centimetres.',False,'That fails the requested unit conversion.')], [('800/10','80'),('1200/10','120')])
    keep('retain_reorder','error_term_change',
        'Use the archived MITx (s,Q) formula. Expected lead-time demand stays 30 and the safety factor stays two. Forecast-error RMSE over lead time falls from five to three. No demand-standard-deviation substitution is made.',[
        ('The reorder point falls from 40 to 36.',True,'The formula gives 30+2*5 and 30+2*3.'),
        ('The reorder point rises because forecast error is smaller.',False,'The positive safety term becomes smaller.'),
        ('The calculated change uses forecast-error RMSE, not an asserted identity with demand variability.',True,'The specified measure is the source error term.'),
        ('The formula sets the order quantity equal to the new reorder point.',False,'The (s,Q) quantity Q is distinct from the trigger s.')], [('30+2*5','40'),('30+2*3','36')])
    keep('retain_tkm','territorial_partition',
        'A constant two-tonne rail consignment travels 30 actual network kilometres in country A and 20 in country B, with no overlapping distance. Apply the cited territorial tonne-kilometre reporting rule.',[
        ('A counts 60 tonne-kilometres for its territory.',True,'Two tonnes times 30 kilometres.'),
        ('B counts 40 tonne-kilometres for its territory.',True,'Two tonnes times 20 kilometres.'),
        ('Each country should count the whole 100 tonne-kilometre journey.',False,'That double-counts foreign distance contrary to the territorial rule.'),
        ('The supplied actual network distances must be replaced by unknown charged distances.',False,'The fallback applies when actual distance is unavailable.')], [('2*30','60'),('2*20','40')])
    keep('retain_capital','stock_flow_boundary',
        'An inland-waterway asset report lists an estimated current infrastructure value and separately the amount spent this year on maintenance. The valuation team can account for depreciation. Use the cited IWT capital-stock scope.',[
        ('Capital stock concerns the current monetary value of physical infrastructure assets.',True,'That is the cited stock concept.'),
        ('Annual maintenance spending is automatically identical to the current asset stock value.',False,'A period expenditure is not the defined asset-stock value.'),
        ('The cited statistical recommendation favors net capital value accounting for depreciation.',True,'The source explicitly recommends that measure.'),
        ('The source mandates ignoring depreciation whenever maintenance occurs.',False,'It gives the opposite recommendation for net value.')])
    keep('retain_outgoing_rail','origin_transit_distinction',
        'Use the 2019 outgoing international rail-goods definition for declaring country A. Consignment X is loaded onto rail in A and unloaded in B. Consignment Y is loaded in B, passes through A without loading or unloading there, and is unloaded in C.',[
        ('X qualifies as outgoing international rail goods from A.',True,'Its rail loading is in A and unloading is abroad.'),
        ('Y qualifies as outgoing goods from A solely because it crosses A.',False,'Through transit is excluded.'),
        ('Loading location matters to the cited classification.',True,'The definition identifies the declaring-country loading location.'),
        ('The definition treats transit throughout as the same as local-origin outgoing goods.',False,'It expressly excludes through transit.')])
    keep('retain_aircraft_moment','signed_moment',
        'In a purely illustrative weight-and-balance calculation using the archived FAA sign convention, one item weighs two units at arm -3 and another weighs one unit at arm +6. These are all items in the example; units are consistent. This is not an aircraft loading plan.',[
        ('The item moments sum to zero.',True,'Two times minus three plus one times six equals zero.'),
        ('The combined center-of-gravity arm in this example is zero.',True,'Total moment zero divided by weight three is zero.'),
        ('Forward and aft arms must both be made positive before summing.',False,'The cited convention uses signed arms.'),
        ('Dividing a moment by a stated index scale changes the physical weight.',False,'An index is a scaled representation, not a weight change.')], [('2*(-3)+1*6','0'),('(2*(-3)+1*6)/(2+1)','0')])
    keep('retain_postponement','allocation_after_information',
        'A generic semi-finished stock can still be assigned to final products A or B. Demand information arrives tomorrow, and the explicitly sufficient customization and delivery time remains after that date. An alternative irreversibly customizes all stock to A today.',[
        ('Waiting for the information preserves allocation flexibility under the supplied timing premise.',True,'The common product has not yet been committed and timing remains sufficient.'),
        ('Customizing everything to A preserves the same flexibility to B.',False,'Customization is stated to be irreversible.'),
        ('The conclusion requires considering remaining lead time.',True,'The source makes postponement benefits conditional on production lead times.'),
        ('This proves waiting is best under every possible lead-time and demand condition.',False,'The example deliberately supplies particular favorable timing conditions.')])
    keep('retain_packages','aggregation_and_roles',
        'Using the archived transport terminology, a sender prepares filled boxes for transport, secures several together on a pallet, and contracts with a carrier. A named recipient is the consignee. Distinguish physical aggregation from contractual roles.',[
        ('Each filled box is a package when prepared for transport.',True,'A package is packaging together with its contents.'),
        ('The grouped packages can form an overpack.',True,'An overpack can group packages on a pallet.'),
        ('The contracting consignor can also act as shipper.',True,'The cited roles permit this overlap when the consignor contracts with the carrier.'),
        ('Grouping boxes physically proves the recipient undertakes carriage.',False,'Aggregation does not assign the carrier role to the consignee.')])
    keep('retain_contracts','missing_contract_dimension',
        'Use Rodrigue and Slack\'s cited terminology. An unchanged container moves from a ship to a train. The record omits whether the journey uses one through contract or separate contracts. Assess what the physical record alone establishes.',[
        ('Two transport modes are involved.',True,'Ship and train are different modes.'),
        ('The physical record alone identifies the contractual integration.',False,'The relevant contract fact is expressly missing.'),
        ('An unchanged load unit can facilitate mode transfer without item-by-item handling.',True,'This is the cited physical role of the unit.'),
        ('The movement must be transmodal because the container remains the same.',False,'Transmodal refers to services within the same mode, not the same load unit.')])
    keep('retain_belly','configuration_vs_cargo',
        'A hypothetical aircraft is configured for passengers and their baggage, and carries mail in its belly cargo hold on one flight. Apply the archived passenger-aircraft description only.',[
        ('Carrying that mail is compatible with the cited passenger-aircraft description.',True,'The source permits freight including mail in belly holds.'),
        ('Any mail automatically means the aircraft is configured only as a freighter.',False,'Passenger configuration can carry belly freight.'),
        ('The described mail location matches the source\'s general freight location.',True,'It identifies belly cargo holds.'),
        ('The description proves unlimited cargo capacity.',False,'It describes configuration and location, not an unlimited capacity.')])
    keep('retain_payload','fixed_limit_fuel_tradeoff',
        'In an illustrative aircraft mass-budget comparison, the applicable total mass limit and every nonfuel operating mass stay fixed. A longer trip requires two additional mass units of fuel. Only revenue cargo can be reduced to offset that change; no operating decision is requested.',[
        ('Available revenue-cargo payload falls by two mass units under these premises.',True,'The extra fuel consumes two units of the fixed remaining weight budget.'),
        ('Fuel is automatically revenue payload in the cited distinction.',False,'Payload is the revenue passenger/cargo portion, not all useful load.'),
        ('Useful load and payload must be distinguished when interpreting the comparison.',True,'Fuel and other operating loads also consume capacity.'),
        ('The same payload change follows for every real aircraft without checking applicable limits.',False,'The conclusion uses explicitly fixed limits and other masses.')])
    keep('retain_air_cargo','carried_or_planned',
        'Under the archived broad air-cargo definition, consignment A is already aboard an aircraft and consignment B is property scheduled to be carried aboard one. No special mail, baggage or revenue-subcategory exclusions are specified in this question.',[
        ('A is within the stated carried-property scope.',True,'The definition includes property carried in an aircraft.'),
        ('B is within the stated to-be-carried-property scope.',True,'The definition also includes property to be carried.'),
        ('B is excluded solely because loading has not happened yet.',False,'The definition explicitly includes to-be-carried property.'),
        ('This broad definition alone resolves every narrower statistical cargo subcategory.',False,'It supplies no such additional subcategory rules.')])
