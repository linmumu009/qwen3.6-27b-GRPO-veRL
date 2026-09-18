"""Measurement, coordination and cost-boundary tasks from archived source rules."""

def add_planning(add):
    u='metric_denominators'
    add(u,'train','pooled_line_accuracy',
        'Disjoint teams use the same picking-line accuracy definition. Team A correctly picks 9 of 10 lines; B correctly picks 72 of 90. Every line belongs to exactly one team. Compute the combined line-based accuracy.',[
        ('The combined accuracy is 81 percent.',True,'There are 81 correct lines out of 100.'),
        ('The combined accuracy is 85 percent because the two percentages have equal weight.',False,'The group denominators differ.'),
        ('The correct denominator is two teams.',False,'The requested metric is line-based.'),
        ('The combined accuracy is 90 percent because A is more accurate.',False,'The pooled result includes B as well.')], [('(9+72)/(10+90)','81/100')])
    add(u,'train','locations_vs_periods',
        'A facility reports 90 discrepancy-free locations among 100 inspected locations this quarter, and two discrepancy-free months among the three months. Use the archived KPI definitions and keep the two aggregation bases explicit.',[
        ('The location-based discrepancy-free rate is 90 percent.',True,'Its denominator is inspected locations.'),
        ('The month-based rate equals 90 percent because it covers the same quarter.',False,'It is two thirds, using months.'),
        ('The two rates measure different aggregation bases.',True,'The source permits location or period formulations with explicit definitions.'),
        ('The rates can be pooled by adding 90 locations and two months as interchangeable successes.',False,'The denominator units do not match.')], [('90/100','9/10'),('2/3','2/3')])
    add(u,'train','processing_time_boundary',
        'Order A is received at 08:00, picked by 09:00 and shipped at 11:00. Order B is received at 10:00, picked by 10:30 and shipped at 12:00. These are the only two orders and times are same-day local times.',[
        ('A has a three-hour warehouse order processing time under the cited definition.',True,'The definition runs from receipt to shipment.'),
        ('The two-order average is 2.5 hours.',True,'Receipt-to-shipment durations are three and two hours.'),
        ('Picking completion is the stated endpoint for this KPI.',False,'The source uses actual shipment, not picking completion.'),
        ('A picking-time metric would need a separately stated boundary.',True,'It would measure a different interval.')], [('(3+2)/2','5/2')])
    add(u,'train','consistent_rate_units',
        'A dashboard names four metrics with explicit denominators: correctly put-away items / all put-away items; correctly picked lines / all picked lines; used storage space / available storage space; discrepancy-free locations / inspected locations. Each numerator is measured within its own denominator set.',[
        ('The first definition is consistent with put-away accuracy.',True,'The source measures items put in the correct location.'),
        ('The second is a permitted line-based picking-accuracy definition.',True,'The source allows picking accuracy over items or lines.'),
        ('The third is consistent with storage-space utilization.',True,'It compares used with available storage space.'),
        ('The fourth is a location-based inventory-accuracy definition.',True,'It measures locations without discrepancies.')])
    add(u,'dev','overlapping_audits',
        'Two audits use the same item-accuracy definition. Each reports 8 correct among 10 items. Five items were inspected in both audits, but the report omits which overlapping items were correct. Treat correctness as stable across the audits.',[
        ('There are 15 distinct inspected items.',True,'The union has 10+10-5 items.'),
        ('The distinct-item accuracy is necessarily 16/20.',False,'That pools duplicated observations rather than distinct items.'),
        ('The exact distinct-item correct count is not identified without overlap outcomes.',True,'Correct items counted twice must be removed, and their number is missing.'),
        ('The weighted-rate rule for disjoint groups directly resolves this overlap without more data.',False,'Its disjointness condition is not satisfied.')], [('10+10-5','15')])
    add(u,'dev','inverse_rate_target',
        'A fixed batch has 40 inspected picking lines, 34 correct. A planned audit adds exactly 10 different lines under the same accuracy definition. The objective is combined accuracy at least 88 percent, with no changes to the first batch.',[
        ('All ten added lines must be correct to meet the target.',True,'At least 44 of 50 must be correct, requiring ten more.'),
        ('Nine correct added lines suffice.',False,'43/50 is 86 percent.'),
        ('The batches should receive equal weight because there are two audits.',False,'Their denominators are 40 and ten.'),
        ('The target can be reached with eight correct added lines.',False,'42/50 is 84 percent.')], [('50*88/100-34','10'),('(34+9)/50','86/100')])

    u='mrp_coordination'
    add(u,'train','bom_shared_component',
        'A hypothetical product needs two units of component C and one assembly S. Each S itself needs three additional units of C. These requirements are nonoverlapping, all yields are one, and no stock or scheduled receipts exist. Five products are required.',[
        ('The total gross requirement for C is 25 units.',True,'Each product needs two direct plus three through S, so five times five.'),
        ('The requirement is ten because subassemblies are excluded from a BOM.',False,'A BOM includes parts and subassemblies and their quantities.'),
        ('The requirement is 15 because direct components must be ignored.',False,'Both nonoverlapping levels contribute.'),
        ('Known BOM quantities alone guarantee an uncertainty-free supply system.',False,'Basic MRP does not inherently resolve uncertainty.')], [('5*(2+3)','25')])
    add(u,'train','lead_time_release',
        'A component must be available at the start of day 12. Its known lead time is four full days, measured from release to availability with no nonworking days or other offsets. No existing stock can satisfy the requirement.',[
        ('Release at the start of day 8 gives availability at the start of day 12.',True,'The supplied timing rule adds four days.'),
        ('Release at the start of day 12 satisfies the same requirement.',False,'It arrives four days later.'),
        ('Adding one day to the known lead time requires release one day earlier for the same due date.',True,'The release offset must increase.'),
        ('The source guarantees lead times are known in every real system.',False,'Known lead time is a basic-model assumption, not a universal fact.')], [('12-4','8'),('12-5','7')])
    add(u,'train','joint_cost_tradeoff',
        'Two fully feasible coordination plans have hypothetical costs: plan A costs the firm 80 and the supplier 50; plan B costs the firm 90 and the supplier 20. The only objective is minimizing their combined cost, and these are all included costs.',[
        ('B has the lower combined cost.',True,'B totals 110 versus A at 130.'),
        ('B increases the firm cost.',True,'The firm pays 90 rather than 80.'),
        ('A joint improvement is impossible whenever one party costs more.',False,'The source permits one cost to rise while the total falls.'),
        ('Choosing A by firm cost alone can differ from the joint decision.',True,'Firm-only cost favors A, combined cost favors B.')], [('80+50','130'),('90+20','110')])
    add(u,'train','coordination_scope',
        'A planning team lists product components and quantities, schedules releases using supplied lead times, and compares firm-first supplier-later planning with a plan considering both parties together. It has not modeled uncertainty.',[
        ('Component quantities are within BOM scope.',True,'The source defines BOM contents to include quantities of parts and assemblies.'),
        ('Known lead-time offsets relate releases to required availability.',True,'This is the stated timing relationship.'),
        ('Firm-first then supplier planning is a sequential approach.',True,'The source describes that order as sequential optimization.'),
        ('The setup does not establish that uncertainty has been resolved.',True,'Basic MRP does not inherently solve uncertainty.')])
    add(u,'dev','receipt_netting_and_timing',
        'A scenario requires 18 units of a component at the start of day 10. Six usable units are already in stock and four more arrive at the start of day 9. Treat both as available for this requirement, with no other commitments. New orders arrive three days after release, with unit yield.',[
        ('A new order of eight units covers the remaining requirement.',True,'18 minus six minus four equals eight.'),
        ('Releasing that order at the start of day 7 meets the due time.',True,'Three days of lead time gives day 10.'),
        ('The supplied stock and scheduled receipt reduce the new-order requirement.',True,'Both are explicitly available for this requirement.'),
        ('A release at the start of day 9 still meets the stated due time.',False,'It arrives on day 12.')], [('18-6-4','8'),('10-3','7')])
    add(u,'dev','cost_transfer_identifiability',
        'A claimed joint plan lowers the firm cost from 100 to 85. The supplier cost change is unknown. No other costs exist. Determine what additional fact is needed to establish a strict reduction in combined cost.',[
        ('The supplier cost increase, if any, must be less than 15.',True,'The firm saves 15; a smaller increase preserves a strict total saving.'),
        ('A firm saving alone does not prove a combined saving.',True,'An unknown supplier increase could exceed it.'),
        ('A supplier increase of exactly 15 leaves the combined total unchanged.',True,'It offsets the firm saving exactly.'),
        ('A supplier increase of 20 makes the combined total rise by five.',True,'It exceeds the firm saving by five.')], [('100-85','15'),('-15+20','5')])

    u='network_decisions'
    add(u,'train','locations_without_flows',
        'A network proposal lists two candidate facilities as open but specifies no customer allocation or product flows. Customer demand and facility capacities are known elsewhere but have not been checked against an allocation.',[
        ('A location list alone does not establish a feasible demand-serving flow plan.',True,'Network design concerns both locations and flows.'),
        ('Opening any two facilities automatically proves all demand can be served.',False,'Capacity and feasible allocations still matter.'),
        ('Product flow decisions fall outside network design.',False,'The source includes flows explicitly.'),
        ('The proposal already proves minimum cost.',False,'No complete feasible flow or cost comparison is supplied.')])
    add(u,'train','capacity_and_demand',
        'Two open facilities have hypothetical capacities 7 and 5 units per period. One customer needs 10 units. Either facility may serve it, goods are divisible, and there are no other constraints. Candidate flow X sends 7 and 3; Y sends 8 and 2.',[
        ('X is feasible for the supplied capacities and demand.',True,'It sends ten with neither capacity exceeded.'),
        ('Y is feasible because its total is ten.',False,'Its first facility exceeds capacity seven.'),
        ('Demand totals and each facility limit must both be checked.',True,'An aggregate total alone misses the individual violation.'),
        ('The source fixes these numerical capacities for real facilities.',False,'They are explicit hypothetical scenario parameters.')], [('7+3','10'),('8+2','10')])
    add(u,'train','fixed_and_flow_cost',
        'Exactly one facility must open to serve five units. Both candidates have sufficient capacity and all routes are allowed. A has fixed cost 30 and unit flow cost 4; B has fixed cost 10 and unit flow cost 9. These are all costs.',[
        ('Opening A costs 50 in total.',True,'30 plus five times four is 50.'),
        ('Opening B costs 55 in total.',True,'10 plus five times nine is 55.'),
        ('B is preferred solely because its fixed cost is lower.',False,'The specified total objective includes flow costs.'),
        ('A minimizes cost among these two fully specified choices.',True,'50 is below 55.')], [('30+5*4','50'),('10+5*9','55')])
    add(u,'train','model_components',
        'A hypothetical network model chooses which facilities operate and how much product travels along each allowed link. It then checks capacity and demand constraints and sums explicitly supplied fixed and flow costs.',[
        ('Facility operation is a location decision.',True,'The source includes selecting operating locations.'),
        ('Link quantities are product-flow decisions.',True,'The source includes the flows through the network.'),
        ('A cost claim must be evaluated over feasible choices under the supplied model.',True,'The scenario imposes capacity and demand constraints.'),
        ('A low-cost location can still be unusable for a given flow if a stated constraint forbids it.',True,'Costs do not remove feasibility conditions.')])
    add(u,'dev','cut_capacity',
        'Facilities A and B each have capacity six. Customers X and Y require seven and three units respectively. Only A may serve X; either may serve Y. Goods are divisible and all other conditions are unrestricted. Total supply capacity is twelve.',[
        ('The network cannot satisfy X under the stated links.',True,'Its sole eligible supplier can deliver at most six, below seven.'),
        ('Twelve total capacity proves feasibility for ten total demand.',False,'The route restriction creates a local shortfall.'),
        ('Adding an allowed B-to-X link can remove this particular shortfall.',True,'For example A sends six to X and B sends one to X plus three to Y.'),
        ('Lowering flow costs alone cannot remove this capacity-and-link shortfall.',True,'The infeasibility is structural, not a cost level.')], [('7+3','10'),('1+3','4')])
    add(u,'dev','opening_condition_counterfactual',
        'A feasible plan sends four units from an open facility K to its only customer. A proposed change closes K while preserving that positive flow. The explicit model rule permits outbound flow only from open facilities; no replacement facility or flow is proposed.',[
        ('The changed plan violates the stated opening-flow rule.',True,'It keeps positive flow from a closed facility.'),
        ('The scenario couples location and flow decisions.',True,'Positive outbound flow requires an open facility.'),
        ('The original feasibility does not automatically transfer to the changed plan.',True,'The changed opening decision violates a constraint.'),
        ('Checking only the customer flow total would miss this opening violation.',True,'The total stays four while the facility is closed.')])

    u='outsourcing_governance'
    add(u,'train','delegation_boundary',
        'A public-health organization contracts warehousing execution to a 3PL. Its manager proposes eliminating all internal contract management and KPI monitoring because the provider now performs the physical work. Apply the archived health-commodity outsourcing guidance.',[
        ('The proposal conflicts with the retained management responsibility.',True,'The guidance requires continued involvement, KPI monitoring and contract skills.'),
        ('Outsourcing physical work automatically removes supply-chain management responsibility.',False,'The source explicitly rejects that conclusion.'),
        ('Signing a contract proves the provider is reliable forever.',False,'Provider reliability requires assessment and monitoring.'),
        ('A 3PL contract guarantees lower cost for every organization.',False,'The source presents a contingent decision, not a universal saving.')])
    add(u,'train','provider_and_oversight',
        'A proposed provider has a low quoted price but no assessed reliability record. The organization also lacks staff able to manage the contract. Under the cited framework, evaluate these two unresolved issues.',[
        ('Provider availability and reliability deserve assessment.',True,'The source calls them key initial questions.'),
        ('The lowest quote alone resolves all outsourcing concerns.',False,'It does not establish quality, reliability or management capability.'),
        ('Contract-management capability may need to be developed.',True,'The organization must possess or develop the relevant skills.'),
        ('The framework says outsourcing must be rejected in every developing-country setting.',False,'It describes successful well-managed deployments and conditional assessment.')])
    add(u,'train','explicit_total_cost',
        'A hypothetical in-house operation costs 100 per period. A feasible outsourcing proposal charges 82 and requires retained contract-management cost 12 plus monitoring cost 9. These categories do not overlap and are the only included costs.',[
        ('The outsourcing included total is 103.',True,'82+12+9 equals 103.'),
        ('The provider charge alone is below the in-house total.',True,'82 is below 100.'),
        ('The proposal lowers the specified included total.',False,'103 exceeds 100.'),
        ('Retained management costs can affect the outsourcing comparison.',True,'Outsourcing preserves management responsibilities, and costs are explicitly supplied.')], [('82+12+9','103')])
    add(u,'train','managed_external_execution',
        'An organization delegates transport execution to an external provider, retains supply-chain management, trains contract staff, and monitors performance indicators. It treats potential savings as a claim to be tested.',[
        ('External execution fits the cited outsourcing concept.',True,'Previously internal functions are performed by an external provider.'),
        ('Retaining management is consistent with the guidance.',True,'Responsibility is not removed by outsourcing.'),
        ('Developing contract skills is consistent with the guidance.',True,'The required skills may need to be developed.'),
        ('Testing savings is preferable to assuming a universal saving from the definition.',True,'The source makes benefits conditional on the situation.')])
    add(u,'dev','mixed_kpi_interpretation',
        'After outsourcing, on-time delivery improves while product-damage incidents rise. The contract contains separate delivery and damage targets, and the organization retains monitoring responsibility. No weights for combining the targets are specified.',[
        ('Delivery improvement alone does not establish improvement on both targets.',True,'Damage performance moved in the opposite direction.'),
        ('The organization should ignore the damage metric because execution is external.',False,'It retains monitoring responsibility.'),
        ('The conflicting outcomes warrant examining the separate indicators and contract terms.',True,'The guidance calls for KPI monitoring and contract management.'),
        ('The missing aggregation weights prevent calculation of a uniquely defined weighted overall score.',True,'The aggregation rule is not supplied.')])
    add(u,'dev','handover_responsibility',
        'A first provider is replaced by a second. During the transition, the organization retains contract authority, but both providers claim that the other will supply performance data. The transition agreement has not assigned that reporting duty.',[
        ('Retained management requires resolving the reporting arrangement rather than assuming responsibility vanished.',True,'The organization must maintain involvement and performance monitoring.'),
        ('Changing providers does not cancel the organization\'s management responsibility.',True,'Provider identity does not remove the retained duty.'),
        ('The archived source cannot identify the reporting provider under this unspecified agreement.',True,'The source does not supply a missing scenario contract term.'),
        ('Two providers being involved does not itself resolve the reporting gap.',True,'The scenario expressly leaves the duty unassigned.')])

    u='holding_cost'
    add(u,'train','rate_boundary',
        'An inventory calculation uses capital cost 8, operating cost 5 and expected obsolescence loss 3 per period. They are disjoint categories within the stated holding-cost boundary, with no other included costs.',[
        ('The included holding cost is 16.',True,'8+5+3 equals 16.'),
        ('Obsolescence must always be excluded from holding cost.',False,'The source includes such inventory risks.'),
        ('The source requires replacing the supplied costs by a universal 25 percent rate.',False,'Any fixed percentage is a scenario assumption, not an identity.'),
        ('Capital opportunity cost is outside every possible holding-cost boundary.',False,'It is a named component in the source.')], [('8+5+3','16')])
    add(u,'train','overlapping_operating_charge',
        'A ledger has holding cost 30, explicitly including a storage charge of 7. It separately adds that same storage charge of 7 and a distinct transport charge of 9 to form a combined total. Keep every real cost once.',[
        ('The ledger duplicates the storage charge.',True,'The same seven is inside holding cost and added separately.'),
        ('The corrected combined total is 46.',False,'That retains the duplicate.'),
        ('The corrected combined total is 39.',True,'30 already includes storage, so only the distinct nine is added.'),
        ('All transport charges must be dropped because one storage charge overlaps.',False,'The transport charge is expressly distinct.')], [('30+9','39'),('30+7+9','46')])
    add(u,'train','different_costing_purposes',
        'For a hypothetical action comparison, a warehouse rent of 50 is fixed under either action. Inventory risk cost is 6 under A and 2 under B. A full allocated report includes rent; an incremental comparison includes only costs that change.',[
        ('The full allocated totals can include the rent.',True,'The stated reporting boundary includes it.'),
        ('The incremental difference from A to B is a saving of four.',True,'The unchanged rent cancels, while risk falls from six to two.'),
        ('Rent has become physically zero because it cancels in the comparison.',False,'An unchanged expense still exists.'),
        ('Reporting purpose and costing boundary must be distinguished.',True,'The source warns that included costs depend on the boundary.')], [('6-2','4')])
    add(u,'train','component_scope',
        'A source-based inventory review separately lists capital opportunity cost, operating expenses, deterioration risk and shrinkage risk. The values are item-specific and the review checks for overlapping entries.',[
        ('Capital opportunity cost can belong in holding cost.',True,'It is expressly included by the source.'),
        ('Operating expenses can belong in holding cost.',True,'They are a named component.'),
        ('Deterioration and shrinkage can be relevant risk components.',True,'Both are among the cited risks.'),
        ('Checking overlaps is consistent with the source guidance.',True,'It warns against counting the same expense twice.')])
    add(u,'dev','inventory_mix_rate',
        'Class A has inventory value 100 and an assumed holding rate of 10 percent for this calculation. Class B has value 300 and an assumed rate of 20 percent. Rates share the same period and boundary, and costs are additive.',[
        ('Total holding cost is 70.',True,'A contributes ten and B contributes sixty.'),
        ('The combined effective rate is 15 percent by unweighted averaging.',False,'The inventory values differ; the effective rate is 17.5 percent.'),
        ('The combined effective rate is 17.5 percent.',True,'70 divided by 400 equals 0.175.'),
        ('The supplied rates remain calculation assumptions rather than universal inventory rates.',True,'Their scope is explicitly limited to this calculation.')], [('100*10/100+300*20/100','70'),('70/400','175/1000')])
    add(u,'dev','break_even_overlapping_fee',
        'Action B reduces an already consolidated holding-cost total by 12 compared with A. Implementing B costs a distinct one-time fee of 8 over the same comparison horizon. A proposed extra handling saving of 5 is already included in the reduction of 12.',[
        ('The net saving from B is four.',True,'The nonduplicated reduction of 12 minus the separate fee of eight equals four.'),
        ('Reporting a saving of nine by adding handling again would double-count five.',True,'That five is already included in the reduction of twelve.'),
        ('The explicitly distinct fee must be included separately in this comparison.',True,'It is not part of the consolidated holding reduction.'),
        ('B still has a positive net saving despite its implementation fee.',True,'The fee is smaller than the nonduplicated saving.')], [('12-8','4')])
