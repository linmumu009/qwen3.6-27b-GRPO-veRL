"""Source-authored applications, with scenario constructions reserved by split."""

def add_warehouse(add):
    u='warehouse_roles'
    add(u,'train','two_stream_assignment',
        'A site receives finished goods from three factories, groups them by retail-store destination, and sends consolidated store replenishments. A separate site sends individual online orders to households. Classify only the operations described.',[
        ('The first operation demonstrates a distribution-center function.',True,'Multiple sources are consolidated for common destinations.'),
        ('The second operation is necessarily raw-material storage.',False,'Household order fulfillment does not establish raw materials.'),
        ('Consolidation proves the first site handles only unfinished assemblies.',False,'The premises expressly identify finished goods.'),
        ('Both operations must be identical because both ship goods.',False,'The customer and consolidation functions differ.')])
    add(u,'train','manufacturing_buffers',
        'At a manufacturer, area R stores purchased feedstock, area W stores unfinished assemblies, and area F buffers completed output next to the production line. Assess these proposed classifications.',[
        ('R can perform raw-material storage.',True,'Feedstock awaiting production fits the raw-material function.'),
        ('F is a local warehouse solely because it is near the manufacturing line.',False,'Local warehouses are near customers for rapid response, not defined by production proximity.'),
        ('W can perform a WIP warehouse function.',True,'Its inventory consists of partially completed assemblies.'),
        ('Finished-goods buffering and WIP storage are the same material-state category.',False,'Completed and partially completed stock differ.')])
    add(u,'train','crossdock_scope',
        'A terminal receives palletized materials from several suppliers and cross-docks mixed pallet shipments. It does not assemble products or unpack individual consumer orders. Which statements follow from the MITx function distinctions?',[
        ('The described flow is consistent with a mixing-center function.',True,'The source associates mixing with multiple origins and mixed-material cross-docking.'),
        ('The record establishes a WIP function.',False,'No unfinished assemblies are specified.'),
        ('The absence of individual consumer orders means fulfillment is not established by this record.',True,'That distinct function needs consumer-order evidence.'),
        ('Several supplier origins alone do not prove that manufacturing occurs at the terminal.',True,'Origins describe inbound flows, not production.')])
    add(u,'train','classification_dimensions',
        'One campus contains feedstock storage, unfinished-assembly storage, completed-output buffering near production, and a separate stockholding depot near distant customers for rapid response. Treat these as four expressly described operations.',[
        ('Feedstock storage is consistent with raw-material storage.',True,'It stores production inputs.'),
        ('Unfinished-assembly storage is consistent with WIP storage.',True,'The assemblies are partially completed.'),
        ('Completed-output buffering near production is consistent with finished-goods warehousing.',True,'Both material state and location match.'),
        ('The distant customer-serving depot is consistent with a local-warehouse function.',True,'It is near customers to support rapid response.')])
    add(u,'dev','missing_activity_evidence',
        'A property register says only that Site K is a large building beside a motorway. It gives no stock state, customer type, consolidation process, or order size. An analyst labels it a fulfillment center. Evaluate what the record supports.',[
        ('The building description alone establishes individual-consumer order fulfillment.',False,'No consumer-order activity is specified.'),
        ('Customer type and order-flow evidence would help test that classification.',True,'Those activities distinguish fulfillment from other functions.'),
        ('Road proximity establishes that every item is raw material.',False,'Location does not establish material state.'),
        ('The activity-based classification remains underdetermined.',True,'Several function-defining facts are missing.')])
    add(u,'dev','function_change_over_time',
        'In period A, a facility buffers finished products beside the factory. In period B, that activity ends and it instead receives goods from several origins, combining them into common-destination shipments. No consumer-order operation is specified.',[
        ('The activity evidence changes from finished-goods buffering to distribution-center consolidation.',True,'The two periods describe different functions.'),
        ('The unchanged address forces the functional classification to remain unchanged.',False,'Classification depends on operations, not address alone.'),
        ('Period B necessarily demonstrates e-commerce fulfillment.',False,'Individual-consumer orders are not specified.'),
        ('Period A demonstrates incomplete assembly storage.',False,'The premises explicitly say finished products.')])

    u='handling_granularity'
    add(u,'train','same_throughput',
        'Two hypothetical methods move the same 240 eaches. Method P moves ten intact pallets of 24 eaches; method E moves each item separately. No damage, transport or packaging cost figures are supplied. Use the source handling-unit tendency without inventing a cost ratio.',[
        ('The source tendency favors larger handling units for handling cost, all else equal.',True,'Smaller handling units generally incur greater handling cost.'),
        ('The source proves E costs exactly 24 times as much.',False,'It gives a tendency, not a linear universal ratio.'),
        ('Ten pallet moves imply only ten eaches of throughput.',False,'Each pallet contains 24 eaches.'),
        ('Handling-unit size alone proves the lower total system cost.',False,'Other system costs have not been specified.')], [('10*24','240')])
    add(u,'train','explicit_exception_costs',
        'For a fixed shipment, hypothetical method P has handling cost 40 and repacking cost 90. Method C has handling cost 70 and repacking cost 10. All amounts use the same currency; these are the only included costs. P uses larger handling units.',[
        ('P has the lower handling cost.',True,'40 is below 70.'),
        ('P must have the lower included total because its units are larger.',False,'P totals 130 and C totals 80.'),
        ('C has the lower included total.',True,'Its higher handling cost is outweighed by lower repacking cost.'),
        ('The comparison disproves every tendency relating unit size to handling cost.',False,'The stated handling costs actually follow that tendency; total cost differs.')], [('40+90','130'),('70+10','80')])
    add(u,'train','downstream_design',
        'A warehouse receives pallets and serves one channel in full cases and another in individual eaches. The source describes handling-unit size as a design and operating concern. No claim is made that every downstream flow must become smaller.',[
        ('The two outbound channels can create different handling requirements.',True,'Case and each handling differ in granularity.'),
        ('Pallet receipt alone establishes that all outbound operations are pallet-to-pallet.',False,'The outbound channels are expressly cases and eaches.'),
        ('Package size belongs in operating design considerations.',True,'The source identifies it as affecting design and operations.'),
        ('The usual downstream trend is a tendency rather than a proof about every shipment.',True,'The source uses a general progression, not an exceptionless rule.')])
    add(u,'train','event_count_and_cost',
        'A trial moves 12 intact cases in method C and 144 individual units in method E, each case containing 12 units. Costs per move and all other cost categories are unmeasured. Judge these limited conclusions.',[
        ('Both methods move 144 units of product.',True,'12 cases times 12 units equals 144 units.'),
        ('E records more handling events.',True,'144 individual moves exceed 12 case moves.'),
        ('An exact total-cost ratio cannot be calculated from these records alone.',True,'Per-event and other costs are missing.'),
        ('The source gives a reason to investigate granularity when comparing handling cost.',True,'Its general rule connects smaller handling units to higher handling cost.')], [('12*12','144')])
    add(u,'dev','mixed_granularity_ledger',
        'An order contains 72 units. Plan A moves six cases of 12 units. Plan B moves four such cases intact and 24 remaining units individually. A ledger counts one event per intact case or individual unit; no costs are recorded.',[
        ('Plan B has 28 recorded handling events.',True,'Four case moves plus 24 individual moves total 28.'),
        ('Plan B moves more product units than A.',False,'Both move 72 units.'),
        ('The event-count ratio is automatically the total-cost ratio.',False,'Event counts do not establish monetary cost per event or other costs.'),
        ('Plan A has six recorded handling events.',True,'Its six cases are all moved intact.')], [('4*12+24','72'),('4+24','28'),('6*12','72')])
    add(u,'dev','inverse_cost_identifiability',
        'For the same product quantity, one trial reports lower total cost for each-picking than for pallet handling. It supplies no breakdown between handling, repacking, damage and other costs. What can be concluded about the source handling-unit tendency?',[
        ('The total alone cannot identify which trial had the lower handling-cost component.',True,'A total confounds the unspecified cost components.'),
        ('The lower total proves the each-picking handling component is lower.',False,'Other components could explain the difference.'),
        ('All source statements about handling granularity are logically refuted.',False,'An aggregate total does not directly test the handling component.'),
        ('The missing pallet-handling cost equals zero.',False,'No such value is supplied.')])

    u='shipping_controls'
    add(u,'train','defect_to_control',
        'A shipment has an incorrect destination label. The pallet is stable, damage protection is adequate, and the loading slot is available. Select the action that directly addresses the demonstrated defect under the check/pack/ship distinctions.',[
        ('Verify and correct the shipping label.',True,'Label creation and verification are checking activities.'),
        ('Change only the yard parking assignment.',False,'Yard management does not correct the wrong label.'),
        ('Add wrapping while leaving the label unchanged.',False,'Damage protection does not resolve the label error.'),
        ('Increase staging space while leaving the label unchanged.',False,'Space does not fix destination identification.')])
    add(u,'train','paired_control_failures',
        'An audit finds that shipping labels are accurate, weights have not been verified, and fragile items have no adequate damage protection. Assign the two demonstrated gaps to the source functions.',[
        ('Weight verification belongs to checking.',True,'Checking includes confirmation of weight and cube.'),
        ('The protection gap can be addressed through packing.',True,'Packing includes damage protection.'),
        ('Accurate labels prove checking is complete.',False,'Weight verification is a separate checking task.'),
        ('Changing dock-door allocation alone resolves both gaps.',False,'Dock allocation does not supply verification or protection.')])
    add(u,'train','process_evidence',
        'A team verifies labels and cube, unitizes cases into stable pallets, and then assigns dock doors and optimizes trailer loading. Use the MITx function distinctions; no enterprise-specific mandatory sequence is assumed beyond the described events.',[
        ('Verifying cube is a checking activity.',True,'The source includes weight and cube confirmation in checking.'),
        ('Unitizing pallets is a packing activity.',True,'The source places pallet unitization under packing.'),
        ('Dock-door assignment is a shipping activity.',True,'Shipping includes dock and yard management.'),
        ('Trailer loading optimization proves every label has been verified.',False,'One shipping activity does not establish a different checking task.')])
    add(u,'train','separate_completion_records',
        'Four completed records show label verification, protective packing, yard management, and optimized container loading respectively. Assess the source-based mappings, without inferring completion of unrecorded tasks.',[
        ('Label verification maps to checking.',True,'It is explicitly listed under checking.'),
        ('Protective packing maps to packing.',True,'Damage protection is a packing responsibility.'),
        ('Yard management maps to shipping.',True,'It is listed among shipping activities.'),
        ('Container loading optimization maps to shipping.',True,'Loading optimization is explicitly included.')])
    add(u,'dev','unobserved_completion',
        'A departure log records only that a trailer left its assigned dock on time. An auditor concludes that label, weight, cube and damage-protection checks all passed. There are no other records or guarantees linking departure to those checks.',[
        ('The conclusion about all checks is not established by this log.',True,'Departure evidence alone does not prove separate checking and packing tasks.'),
        ('The source distinctions make departure a substitute for weight verification.',False,'The functions are distinct.'),
        ('Separate check and pack evidence would be needed to substantiate that conclusion.',True,'The claimed tasks are not documented.'),
        ('On-time departure logically proves that the label was wrong.',False,'Absence of checking evidence proves neither correctness nor error.')])
    add(u,'dev','control_after_repack',
        'A consignment was checked and then repacked. The hypothetical operating rule requires renewed weight and cube verification whenever repacking changes either value. Repacking changed both; no renewed verification has occurred. The label remains correct.',[
        ('The specified renewed weight and cube checking is still outstanding.',True,'The explicit trigger occurred and the required verification is absent.'),
        ('A correct label alone satisfies the stated re-verification rule.',False,'That rule concerns changed weight and cube.'),
        ('The general source requires this exact trigger rule in every enterprise.',False,'The trigger is an explicit scenario rule, not a universal source requirement.'),
        ('Packing changes cannot affect any checking requirement in this scenario.',False,'The supplied operating rule directly links them.')])

    u='activity_distribution'
    add(u,'train','same_average_different_peak',
        'Two three-day order profiles are A=(40,40,40) and B=(10,10,100). Each order requires one unit of work. A hypothetical same-day system can process 60 units daily, with no carryover, overtime or advance processing.',[
        ('Only B has a day exceeding the stated capacity.',True,'Its peak is 100 while A peaks at 40.'),
        ('Equal average demand guarantees equal peak demand.',False,'Both average 40 but their peaks differ.'),
        ('A exceeds capacity every day.',False,'40 is below 60.'),
        ('B meets the same-day constraint because its three-day total is only 120.',False,'The 100-unit day exceeds daily capacity regardless of the total.')], [('(40+40+40)/3','40'),('(10+10+100)/3','40')])
    add(u,'train','data_source_selection',
        'A warehouse study needs SKU attributes, timestamps and sizes of customer orders, and the positions of stock in the building. It currently has only a monthly revenue total. Evaluate proposed additions to the source activity profile.',[
        ('Master SKU data can supply item attributes.',True,'The source identifies master SKU data as a profiling input.'),
        ('The monthly revenue total uniquely determines daily order sizes.',False,'Aggregate revenue does not identify the order distribution.'),
        ('Order history and warehouse-location data address the missing activity and position information.',True,'Both are listed profiling sources.'),
        ('Daily variation becomes irrelevant once monthly revenue is known.',False,'The source explicitly requires examining peaks and dips.')])
    add(u,'train','units_and_lines',
        'Profile X has 80 pick-lines daily with one unit per line. Profile Y has 20 lines daily with four units per line. No travel paths, item-handling times or total labor measurements are supplied.',[
        ('Both profiles contain 80 product units daily.',True,'80 times one and 20 times four are equal.'),
        ('X has more pick-lines.',True,'80 lines exceed 20.'),
        ('Equal units prove equal total labor time.',False,'Line count and unmeasured travel/handling can differ.'),
        ('Both lines per day and units per line belong in an activity profile.',True,'The source lists both dimensions.')], [('80*1','80'),('20*4','80')])
    add(u,'train','profile_dimensions',
        'A redesign team records the number of SKUs, daily pick-lines and units per line, the size of shipped orders, and the introduction rate and lifecycle of new SKUs. It retains daily distributions as well as averages.',[
        ('SKU count is a relevant profile dimension.',True,'It is explicitly listed in the source.'),
        ('Order-size distribution can reveal information absent from a simple average.',True,'Distributions expose peaks and dips.'),
        ('New SKU introductions and lifecycle are relevant.',True,'Both appear in the source activity-profile list.'),
        ('Keeping daily distributions is consistent with the source guidance.',True,'It advises examining distributions, not only averages.')])
    add(u,'dev','inverse_peak_requirement',
        'Four days contain a total of 200 one-unit orders. One day is known to contain 90 orders; the other three counts are nonnegative but unknown. There is no carryover or advance work. Assess what follows for a same-day capacity of 70.',[
        ('Capacity 70 cannot meet the known 90-order day.',True,'The observed peak already exceeds capacity.'),
        ('An average of 50 proves capacity 70 is sufficient on every day.',False,'The known peak contradicts that claim.'),
        ('The remaining three counts must all be equal.',False,'Only their total of 110 is determined.'),
        ('The observed data imply a required daily capacity of at least 90.',True,'Serving the known day requires at least 90; other days could require more.')], [('200/4','50'),('200-90','110')])
    add(u,'dev','aggregation_hides_burst',
        'Two shifts each receive 120 orders over four hours. Shift A receives 30 each hour. Shift B receives all 120 in its final hour. Orders cannot be served before arrival and must finish within their arrival hour. Processing capacity is 40 per hour.',[
        ('A can meet all stated hourly deadlines but B cannot.',True,'A never exceeds 40, while B needs 120 in its final hour.'),
        ('Both are feasible because shift capacity is 160.',False,'Unused earlier capacity cannot serve orders that have not arrived.'),
        ('Equal shift totals establish equal arrival profiles.',False,'The hourly profiles differ.'),
        ('The source recommends discarding hourly variation once shift totals are available.',False,'It recommends examining distributions and peaks.')], [('4*40','160'),('4*30','120')])

    u='storage_tradeoffs'
    add(u,'train','frequency_distance_assignment',
        'In the cited simplified slotting heuristic, two interchangeable locations have receipt-through-storage-to-shipping distances 12 and 30. SKU H has 50 visits per period and SKU L has 10. There are no compatibility constraints in this scenario.',[
        ('The heuristic assigns H to distance 12 and L to distance 30.',True,'It pairs greater visit frequency with smaller distance.'),
        ('The heuristic sorts visits and distances in the same ascending order.',False,'Frequency is descending and distance ascending.'),
        ('The source proves this assignment optimal for every possible warehouse layout.',False,'Its scope is a simplified heuristic, not a universal proof.'),
        ('The distance value is itself a count of SKU visits.',False,'Distance and frequency are separate dimensions.')])
    add(u,'train','empty_location_reassignment',
        'Product A has left a reserved location empty while product B needs space. Compare a dedicated policy reserving that location for A with a shared policy allowing reassignment after emptying. Assume B physically fits and no other constraint prevents its use.',[
        ('The shared policy permits assigning the empty location to B.',True,'Reassignment after emptying is its defining property.'),
        ('The dedicated policy automatically gives B the same permission.',False,'It continues reserving the location for A.'),
        ('Shared storage requires reliable location information for disciplined putaway and picking.',True,'The source identifies those operating requirements.'),
        ('Shared storage guarantees lower labor for every possible inventory pattern.',False,'The source describes a space/labor trade-off, not a universal labor reduction.')])
    add(u,'train','space_vs_nearest',
        'The same SKU occupies a near location with 20 units and a far location with three units. An order requires three units. Both locations are accessible. The objective is either minimizing this pick travel or emptying a location now; no tie is present.',[
        ('Picking all three from the far location empties it.',True,'It contains exactly the order quantity.'),
        ('Picking from the near location empties it.',False,'It would retain 17 units.'),
        ('The travel and space-release objectives can favor different choices.',True,'The near location minimizes travel while the far location becomes empty.'),
        ('The source treats these objectives as potentially conflicting.',True,'It explicitly contrasts nearest picking with freeing a partly filled location.')], [('20-3','17')])
    add(u,'train','idealized_utilization_scope',
        'Use only the book idealization with constant demand, equal locations and one active pick location. It gives mean utilization 1-H_k/(2k), with H_k the sum of reciprocal integers from 1 to k. This model is not claimed to describe every real warehouse.',[
        ('For k=1 the model gives 50 percent.',True,'One minus one half equals one half.'),
        ('For k=2 the model gives 62.5 percent.',True,'H_2=1.5 and 1-1.5/4=0.625.'),
        ('The specified assumptions are part of the result scope.',True,'The source explicitly limits this formula to its idealization.'),
        ('The formula alone cannot certify an actual warehouse occupancy rate outside that scope.',True,'Applicability requires the stated assumptions.')], [('1-1/2','1/2'),('1-(1+1/2)/4','5/8')])
    add(u,'dev','compatibility_breaks_universal_claim',
        'SKU H is visited 50 times and L ten times. A near location is legally usable in this hypothetical scenario only for L; a far location can hold either. Both SKUs must be stored simultaneously, one per location. Consider an unconditional frequency-distance assignment claim.',[
        ('Assigning H near is infeasible under the supplied compatibility constraint.',True,'That location is explicitly restricted to L.'),
        ('The simple heuristic overrides every scenario constraint.',False,'A ranking heuristic does not waive feasibility constraints.'),
        ('H far and L near is a feasible assignment.',True,'Both compatibility and one-per-location requirements are satisfied.'),
        ('The source proves H near is globally optimal even when it is forbidden.',False,'It does not make a universal optimality claim.')])
    add(u,'dev','joint_order_frees_location',
        'A near location contains two units of a SKU and a far location contains nine. One order requires two units and either location could supply it. Unlike a previous generic trade-off, assess this specific state with travel distance strictly increasing from near to far.',[
        ('Picking the near location both minimizes travel and empties a location.',True,'The near stock equals the order quantity.'),
        ('Travel minimization and space release must conflict in every state.',False,'Here the same action achieves both.'),
        ('Picking two from the far location empties it.',False,'Seven units remain.'),
        ('The source requires choosing the far location whenever any space-release objective exists.',False,'It describes a possible trade-off, not an unconditional action rule.')], [('9-2','7')])
