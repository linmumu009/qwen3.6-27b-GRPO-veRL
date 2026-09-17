"""A fixed source-authored quality pilot. No official questions are inputs.

These are development diagnostics, not independent holdouts or training releases.
All facts come from the frozen source packet or explicit hypothetical premises.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from cpt_composed_probe import arithmetic

PACKET_SHA = 'b42fd573fbb9c740446bd3c9f020f8181967f2ac96b40fb0f27f261677743cde'


def build(packet):
    assert hashlib.sha256(packet.read_bytes()).hexdigest() == PACKET_SHA
    requests = [json.loads(s) for s in packet.read_text(encoding='utf-8').splitlines()]
    by_unit = {r['unit']: r for r in requests}
    rows = []

    def add(unit, split, question, options, answers, reasons, structure, calculations=()):
        request = by_unit['expand-' + unit]
        row = dict(id=f'authored-{unit}-{split}-{sum(r["unit"] == unit and r["split"] == split for r in rows)}',
            unit=unit, split=split, question=question + ' Select all correct statements.',
            options=options, correct_indices=answers, option_reasons=reasons,
            reasoning_requirement=structure, arithmetic_checks=list(calculations),
            sources=request['sources'], authoring_scope=request['authoring_scope'],
            source_request_packet_sha256=PACKET_SHA, author='Codex source-authored operator revision',
            training_allowed=False, purpose='authoring_quality_diagnostic',
            review_status='operator_checked_pending_blind_model_review')
        assert len(options) == len(reasons) == 4 and len(set(options)) == 4
        assert answers == sorted(set(answers)) and all(type(i) is int and 0 <= i < 4 for i in answers)
        for expression, expected in calculations:
            assert arithmetic(expression) == arithmetic(expected)
        rows.append(row)

    add('pick_lines','train',
        'Use the archived Warehouse & Distribution Science picking terminology. One instruction sends a picker to bin R for six loose units of one SKU. Each physical grab retrieves one unit. No other location is visited.',
        ['This instruction entails six pick-lines.','This instruction entails one pick-line and six grabs.','The six grabs establish six different SKUs.','The six grabs establish six location visits.'],[1],
        ['One instruction/location, not one line per grab.','One location instruction can require multiple grabs.','One SKU is explicitly specified.','Only bin R is visited.'],
        'Distinguish location instructions, physical grabs and SKU count in one operation.')
    add('pick_lines','train',
        'Use the archived Warehouse & Distribution Science terminology. Two pick faces each display 24 SKUs across 12 square metres. In one observed shift, face A receives 120 pick-line visits and face B receives 36. Count repeated visits separately. Use picks per square metre for this shift.',
        ['Both faces have the same SKU density.','Face A has higher observed pick density.','Face B has three observed picks per square metre.','Equal SKU density did not produce equal observed pick density in this shift.'],[0,1,2,3],
        ['24/12 equals 2 for both.','120/12 is 10, greater than 36/12.','36/12 equals 3.','The observed densities are 10 and 3; demand matters.'],
        'Separate stored assortment density from realized picking density.', [('24/12','2'),('120/12','10'),('36/12','3')])
    add('pick_lines','dev',
        'Use the archived Warehouse & Distribution Science terminology. Two successive shifts use exactly the same assortment, storage positions and pick-face area. The second shift records more pick-line visits per square metre. A reviewer must identify explanations compatible with all these observations.',
        ['Different orders can change realized pick density without changing SKU density.','The observation proves that additional SKUs were installed.','Each physical grab must have become a separate pick-line.','These aggregate observations uniquely identify the exact individual orders placed in the second shift.'],[0],
        ['Pick density depends on orders.','Assortment is explicitly unchanged.','A line may require multiple grabs.','Many different order mixes could produce the same aggregate visit count.'],
        'Reverse inference from changed observed activity and invariant physical layout; do not infer a unique cause.')

    add('batch_tradeoff','train',
        'Apply the archived Warehouse & Distribution Science batching trade-off. Relative to separate picking, batching a fixed set of medium orders avoids 90 cost units of walking, adds 50 of separation work and adds 25 of space cost. These are all incremental costs; feasibility and service are unchanged. The objective is lower total cost.',
        ['Batching saves 15 cost units.','Batching is cheaper under these assumptions.','Ignoring space cost would still give the correct numerical saving.','These observations prove batching is cheaper for every medium order set.'],[0,1],
        ['90-50-25=15.','Positive net saving with unchanged feasibility.','Ignoring space yields 40, not 15.','The conclusion is conditional on this cost instance.'],
        'Combine avoided walking and both added cost components; preserve conditional scope.', [('90-50-25','15')])
    add('batch_tradeoff','train',
        'Apply the archived Warehouse & Distribution Science batching trade-off. For the same medium orders, a proposed batch saves 70 cost units of walking but adds 45 of sorting and 30 of space cost. No other cost, capacity or service changes occur. Minimize total cost.',
        ['Batching raises total cost by 5.','Separate picking is cheaper for this instance.','Considering sorting alone would incorrectly favor batching.','Medium order size by itself makes batching the cheaper choice.'],[0,1,2],
        ['45+30-70=5.','Batching costs five more.','70-45=25 appears favorable but omits space.','The source makes batching conditional on its trade-off.'],
        'Identify a counterexample to unconditional batching and diagnose an omitted cost.', [('45+30-70','5'),('70-45','25')])
    add('batch_tradeoff','dev',
        'Apply the archived Warehouse & Distribution Science batching trade-off. A medium-order batch saves 80 cost units of walking and adds 50 of sorting plus 40 of space. At most one intervention can be selected. Intervention X cuts sorting cost by 12; Y cuts space cost by 8; Z cuts sorting by 6 and space by 6. Interventions have no other cost or effect. The objective is to make batching strictly cheaper than separate picking.',
        ['X alone achieves the objective.','Y alone achieves the objective.','Z alone achieves the objective.','Selecting no intervention already achieves the objective.'],[0,2],
        ['50-12+40=78, below 80.','50+40-8=82, above 80.','50-6+40-6=78, below 80.','90 is above 80.'],
        'Select feasible interventions under a one-intervention budget, including a joint reduction across two cost components.', [('50-12+40','78'),('50+40-8','82'),('50-6+40-6','78')])

    add('serial_parallel','train',
        'Use the archived Warehouse & Distribution Science distinction between completion time and picker work. An order arrives at minute 0 and must be loaded by minute 23. Two independent picks take 12 and 8 minutes. Serial: one worker performs both then loads for 2 minutes. Parallel: two workers start at 0, then one worker consolidates for 4 minutes and loads for 2. No travel or queues are omitted.',
        ['Serial loading finishes at minute 22.','Parallel loading finishes at minute 18.','Parallel uses 26 person-minutes in total, including consolidation and loading.','Parallel uses less total person-work than serial.'],[0,1,2],
        ['12+8+2=22.','The longer concurrent pick finishes at 12, then 4+2.','12+8+4+2=26.','Serial uses 22; parallel uses 26.'],
        'Compute elapsed completion separately from summed person-work.', [('12+8+2','22'),('12+4+2','18'),('12+8+4+2','26')])
    add('serial_parallel','train',
        'Use the archived Warehouse & Distribution Science picking trade-off. Two independent zone picks take 9 and 7 minutes. Serial: one worker picks both, then spends 1 minute loading. Parallel: two workers begin together; after both finish, one worker consolidates for 6 minutes then loads for 1. No other work or waiting occurs. The order starts at minute 0 and must be loaded by minute 16.',
        ['Serial picking meets the deadline.','Parallel picking meets the deadline.','Parallel uses fewer person-minutes than serial.','Parallel saves one elapsed minute while adding six person-minutes.'],[1,3],
        ['9+7+1=17, too late.','9+6+1=16.','Parallel uses 23 versus 17.','17-16=1 and 23-17=6.'],
        'Apply a deadline while retaining the coordination-versus-work trade-off.', [('9+7+1','17'),('9+6+1','16'),('9+7+6+1','23'),('23-17','6')])
    add('serial_parallel','dev',
        'Use the archived Warehouse & Distribution Science flow-time concept. An event log records order arrival at minute 0, start of two concurrent picks at minute 3, pick completion at minutes 13 and 9, consolidation from 13 to 17, and loading from 17 to 19. Each pick, consolidation and loading uses one worker. Waiting before picking consumes no worker labor.',
        ['Arrival-to-loaded flow time is 19 minutes.','Total recorded person-work is 22 minutes.','Starting the flow-time clock at minute 3 would correctly report the arrival-to-loaded interval.','The two pick durations sum to the elapsed interval from first pick start to last pick finish.'],[0,1],
        ['Flow time includes the initial three-minute wait.','(13-3)+(9-3)+(17-13)+(19-17)=22.','It would omit the initial wait.','Durations sum to 16; concurrent interval is 10.'],
        'Reconstruct flow time and work from asynchronous events including queueing, not two forward action formulas.', [('(13-3)+(9-3)+(17-13)+(19-17)','22'),('13-3','10')])

    add('transport_capacity','train',
        'Use the archived transport-supply discussion. A fixed service has two departures on one day, each with capacity for 100 identical units. At the first departure, 130 units request carriage; 30 cannot be carried. At the second, 20 units request carriage. Requests cannot shift between departures. There is no other service that day.',
        ['Daily capacity is 200 units.','Unused capacity on the second departure does not eliminate the first departure\'s shortage.','Realized carriage is 120 units.','Daily spare capacity and an unmet peak coexist in this example.'],[0,1,2,3],
        ['100+100=200.','The requests cannot shift.','100+20=120.','Thirty unmet at the first, eighty spare at the second.'],
        'Distinguish period-specific capacity, requested demand and realized movement.', [('100+100','200'),('100+20','120'),('130-100','30'),('100-20','80')])
    add('transport_capacity','train',
        'Use the archived transport-supply discussion. Yesterday a scheduled departure had 40 unused spaces. Today a separate scheduled departure has capacity for 100 units and 120 units request carriage. No additional departure, vehicle, rescheduling or capacity change is possible today.',
        ['Today\'s capacity becomes 140 by carrying forward yesterday\'s unused spaces.','All 120 units can be carried today because yesterday had spare capacity.','At least 20 units cannot be carried by today\'s fixed departure.','The observations prove this transport market can never reach equilibrium.'],[2],
        ['An expired service cannot be stored.','Today has only 100 spaces.','120-100=20.','The source explicitly rejects this universal conclusion.'],
        'Apply service perishability while avoiding an unwarranted market-wide inference.', [('120-100','20')])
    add('transport_capacity','dev',
        'Use the archived transport-supply discussion. One day\'s records show 200 total spaces over two departures and 150 realized units. The manager claims there could not have been any rejected requests that day. Which possible completions of the missing records refute that claim while preserving both totals? Every departure below has capacity 100, and requests cannot shift between departures.',
        ['Requests were 120 then 50; carriage was 100 then 50.','Requests were 75 then 75; carriage was 75 then 75.','Requests were 150 then 0; carriage was 100 then 0.','Requests were 50 then 130; carriage was 50 then 100.'],[0,3],
        ['150 carried and 20 requests rejected.','150 carried but no rejected requests.','Only 100 carried, contradicting the observed total.','150 carried and 30 requests rejected.'],
        'Find countermodels compatible with aggregate capacity and realized movement; test hidden peak conditions.', [('100+50','150'),('120-100','20'),('50+100','150'),('130-100','30')])

    add('retain-32','retention',
        'Use the archived safety-stock terminology. Stock A is explicitly reserved to cover forecast errors and uncertain supplier shortfalls. Stock B exactly covers a confirmed order, with no extra units for uncertainty. A planner proposes removing A because confirmed orders would still be covered.',
        ['A serves the safety-stock purpose.','B alone establishes that uncertainty-related stockout risk is fully covered.','Removing A removes the stated extra buffer against uncertain shortfalls.','Holding A guarantees that no stockout can ever occur.'],[0,2],
        ['A is extra stock for uncertainty.','Confirmed demand coverage is not uncertainty coverage.','That is A\'s specified role.','Mitigating risk is not an absolute guarantee.'],
        'Separate known commitments from uncertainty protection and risk mitigation from guarantees.')
    add('retain-93','retention',
        'Use the archived IATA ULD definition. A is an aircraft container. B is an aircraft pallet together with its net. C is only an aircraft pallet, with no net or container. No other components are present. A clerk rejects B solely because it is not a container.',
        ['A meets the cited definition.','B meets the cited definition.','C meets the cited definition as the specified pallet-and-net combination.','The clerk\'s container-only classification wrongly excludes B.'],[0,1,3],
        ['Aircraft container is one allowed form.','Aircraft pallet and net is the other allowed form.','C is missing the net.','ULD is broader than containers alone.'],
        'Apply the disjunction and required conjunction to complete and incomplete configurations.')
    add('retain-198','retention',
        'Use the 2019 transport-statistics glossary\'s passenger-aircraft scope. An aircraft remains configured with passenger seats and baggage accommodation. It carries passengers, their baggage and some commercial freight in its belly hold. A clerk proposes reclassifying it solely because commercial freight is present.',
        ['The given configuration supports passenger-aircraft classification.','The presence of the stated belly freight alone requires rejecting passenger-aircraft classification.','The stated cargo location is consistent with the glossary\'s general description.','The definition establishes that every passenger aircraft can accept any freight weight.'],[0,2],
        ['Classification is based on the given passenger/baggage configuration.','The definition permits freight in the belly.','Belly holds are the stated general location.','No universal payload capacity is given.'],
        'Separate configuration classification from incidental freight carriage and unprovided capacity.')
    add('retain-0','retention',
        'Use only the archived general definition of air cargo, not a regulatory baggage taxonomy. Crate A is booked for an aircraft tomorrow and is now travelling by truck to the airport. Crate B will complete its entire journey by truck and will never be carried in an aircraft. Neither is currently on an aircraft.',
        ['A falls within the definition because its future air carriage is specified.','A is excluded merely because its current leg is by truck.','B falls within the definition merely because it is property in transport.','Both must already be inside an aircraft to decide their status.'],[0],
        ['The definition includes property to be carried in an aircraft.','A ground feeder leg does not negate the specified future air carriage.','B has no present or future aircraft carriage.','The definition includes future carriage.'],
        'Discriminate future air carriage from a ground-only journey with both current locations held fixed.')
    assert len(rows) == 16
    assert Counter(r['split'] for r in rows) == {'train':8, 'dev':4, 'retention':4}
    assert Counter(len(r['correct_indices']) for r in rows if r['split']=='train') == {1:2,2:2,3:2,4:2}
    return rows


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=build(a.packet);a.out.mkdir(exist_ok=False)
    path=a.out/'cases.private.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    summary=dict(cases_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_packet_sha256=PACKET_SHA,
        cases=len(rows),splits=dict(Counter(r['split'] for r in rows)),training_allowed=False,
        planned_review_calls=16,planned_closed_book_calls=32,closed_book_option_orders=2,
        diagnostic_only=True,limit='Small operator-authored development quality pilot; shared source rules across train/dev, no claim of independent holdout, broad retention, or model improvement.')
    (a.out/'manifest.safe.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
