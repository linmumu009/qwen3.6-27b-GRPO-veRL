"""Build cumulative source cards and quarantined drafting requests; no benchmark input.

The historical 205-source snapshot and development partition stay frozen.
New-to-P1 means absent from its domain tasks AND its 135 replay source IDs.
It does not mean absent from pretraining, earlier CPT, or model knowledge.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from prepare_cpt_transfer_screen import SOURCE_SHA, historical_groups, sha
from build_cpt_transfer_pilot import UNITS

# Source positions are stable only under SOURCE_SHA. These are source-authored
# capability assignments, not mappings from benchmark answer keys.
CARDS = [
 ('warehouse_roles',[73],'warehouse_operations','区分原料、在制品、履约、配送与混合中心的任务',
  'Apply the cited warehouse-function distinctions to explicit flows and customers; do not claim that one real facility has only one role.'),
 ('handling_granularity',[74],'warehouse_operations','区分搬运单位粒度与作业成本方向',
  'Treat smaller-unit handling cost as a general source tendency, not an exact universal ratio; hold throughput and other assumptions explicit.'),
 ('pick_lines',[166],'warehouse_operations','区分拣选行、抓取次数、SKU密度及行走需求',
  'Do not equate a pick-line with one grab; distinguish SKU density from pick density and state any numerical workload assumptions.'),
 ('batch_tradeoff',[167],'warehouse_operations','比较批量拣选节省行走与增加分拣空间的代价',
  'Batch only when stated separation and space costs are lower than avoided walking; do not make medium-order batching universally optimal.'),
 ('serial_parallel',[168],'warehouse_operations','比较串行与并行拣选的流转时间和协调成本',
  'Keep flow-time and total person-work separate; serial picking avoids coordination but is not universally faster or cheaper.'),
 ('zone_work',[169],'warehouse_operations','按工作量分工并处理分区平衡、汇合与发运完整性',
  'Preserve shipment integrity and consolidation cost; distinguish work content from elapsed completion time and state all throughput assumptions.'),
 ('shipping_controls',[77],'warehouse_operations','区分复核、包装保护、集装与发运作业',
  'Distinguish checking labels/weight/cube, packing protection/unitization, and shipping dock/yard/loading activities; do not invent a compulsory enterprise SOP.'),
 ('activity_distribution',[78],'warehouse_operations','用需求分布与峰谷补充平均作业量',
  'Use SKU, order and location records; identical means do not establish identical peaks or sufficient capacity.'),
 ('storage_tradeoffs',[86,196],'warehouse_operations','比较固定共享库位、空间释放和行走距离目标',
  'Separate the frequency-distance heuristic from a global optimum; reserve the harmonic utilization formula for its exact stated idealized model.'),
 ('metric_denominators',[91,170],'measurement_planning','按口径和分母计算仓储指标及合并比率',
  'State whether the denominator is items, lines, locations, orders or periods; combine rates only with matching definitions and nonoverlapping groups.'),
 ('mrp_coordination',[2,192],'supply_planning','从物料清单和提前期安排需求并区分顺序与联合协调',
  'Do not claim deterministic MRP resolves uncertainty; include receipts, stock, yields or calendar conventions only when explicitly supplied.'),
 ('network_decisions',[108],'supply_planning','把设施选址和网络流量作为不同决策',
  'A feasible network needs both locations and flows; numerical optimality requires fully supplied costs, capacities and demand, not source-invented values.'),
 ('outsourcing_governance',[171],'business_governance','外包实施后仍承担合同、指标和供应链管理责任',
  'Use the archived health-commodity framework; outsourcing does not remove management responsibility and is not automatically a cost saving.'),
 ('transport_capacity',[194],'transport_operations','区分潜在需求、实现流量、时段容量及调整时滞',
  'Do not equate unused capacity with zero potential demand or aggregate spare capacity with absence of peaks; do not assert equilibrium is impossible.'),
 ('bridge_access',[199],'transport_operations','区分桥下净空、河流水位与开桥时窗',
  'Separate vessel air draft, available bridge clearance and draft below water; state safety margins and opening windows, never invent current local navigation rules.'),
 ('drought_response',[200],'transport_operations','在水深约束下比较减载、替代运输和库存方案',
  'Treat the 2018 Rhine account as historical; more trips need not restore system throughput, and every remedy needs explicit capacity and feasibility assumptions.'),
 ('road_combinations',[59,61,63,116,117,118],'transport_operations','区分载货车、牵引车、挂车、半挂车及组合',
  'Preserve the 2019 statistical classification and coupling/load-support conditions; distinguish complete vehicle combinations from their components.'),
 ('road_capacity',[119,162,164],'transport_operations','区分载重量、容积与最大许可总重',
  'Keep mass and volume separate; legally permissible gross weight includes the vehicle, and numerical capacity requires explicit tare and load assumptions.'),
 ('rail_track_line',[131,132,133,136,137],'transport_operations','区分轨道、线路、正线、侧线及专用侧线',
  'Use the 2019 statistical scope; track counts and railway-line length are different, and private sidings need their source exclusions.'),
 ('forecast_scope',[89,181],'measurement_planning','比较预测误差口径、汇总方式与港口预测范围',
  'Ordinary MAPE is undefined at zero actuals; keep units and aggregation explicit, and historical illustrative TEU factors must not become universal constants.'),
 ('holding_cost',[197],'business_governance','按成本边界合并库存资本、运营和风险费用',
  'Do not double-count costs or treat a fixed carrying-rate percentage as universal; identify explicitly provided costing boundaries.'),
]

# Existing development sources remain excluded from training. This expands
# qualitative checking within that pool; it is not a fresh final holdout.
RETENTION = [
 (28,'business_environment'),(32,'inventory_risk'),(44,'inventory_risk'),
 (93,'cargo_units'),(96,'cargo_units'),(109,'inventory_risk'),
 (115,'transport_measurement'),(127,'capital_stock'),(139,'transport_measurement'),
 (172,'aircraft_operations'),(191,'postponement'),(193,'cargo_service'),
 (195,'cargo_service'),(198,'aircraft_operations'),(201,'aircraft_operations'),(0,'cargo_service'),
]
TRAIN_DESIGNS = ['single_case_application','boundary_pair','explicit_exclusion',
                 'correct_a_report','compare_two_actions','conditional_counterexample']
DEV_DESIGNS = ['dev_expression_ledger','dev_conditions_reverse_inference','dev_conditions_minimal_change']
RETENTION_DESIGNS = ['retention_application','retention_boundary','retention_decision']


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def validate_evidence(rows, evidence):
    by = {r['key']:r for r in evidence}
    if len(by) != len(evidence):
        raise ValueError('duplicate evidence key')
    counts = Counter()
    for row in rows:
        for link in row['evidence']:
            source = by[link['key']]
            if not source['sha256'] == link['sha256'] == hashlib.sha256(source['text'].encode()).hexdigest():
                raise ValueError('evidence hash mismatch')
            if 'excerpt' in link and link['excerpt'] not in source['text']:
                raise ValueError('evidence excerpt missing')
            counts['references'] += 1
            counts['references_without_excerpt'] += 'excerpt' not in link
    return dict(counts)


def prepare(resources, out):
    source = resources/'llin-cpt-sft-search-20260912/source_generation_requests.private.jsonl'
    evidence_path = resources/'llin-knowledge-complete-20260911/core/sources.private.jsonl'
    old_path = resources/'llin-source-reviewed-20260914-01/reviewed_tasks.private.jsonl'
    if sha(source) != SOURCE_SHA:
        raise ValueError('source snapshot changed')
    rows = read(source); evidence = read(evidence_path)
    if len(rows) != 205 or len({r['id'] for r in rows}) != 205:
        raise ValueError('source coverage mismatch')
    evidence_check = validate_evidence(rows, evidence)
    script = Path(__file__).with_name('generate_source_condition_tasks.py')
    groups = historical_groups(rows, script)
    dev_groups = {g for g in groups.values() if int(hashlib.sha256(g.encode()).hexdigest(),16)%5==0}
    dev_topics = {t for r in rows if groups[r['id']] in dev_groups for t in r['topics']}
    replay = [r for r in read(old_path) if r['split']=='train']
    if len(replay) != 135:
        raise ValueError('legacy replay count changed')
    replay_ids = {r['unit_id'] for r in replay}
    p1_ids = {'llin-core-'+s for _, ids in UNITS.values() for s in ids}
    known_ids = p1_ids | replay_ids
    cards = []; requests = []
    for card_index,(name, positions, category, ability, constraint) in enumerate(CARDS):
        selected = [rows[i] for i in positions]
        if any(groups[r['id']] in dev_groups or set(r['topics']) & dev_topics for r in selected):
            raise ValueError('frozen development source/topic selected for expansion')
        card = dict(id='expand-'+name, category=category, ability=ability, source_ids=[r['id'] for r in selected],
            source_groups=sorted({groups[r['id']] for r in selected}),
            source_titles=[r['title'] for r in selected], topics=sorted({t for r in selected for t in r['topics']}),
            new_to_p1_source_ids=[r['id'] for r in selected if r['id'] not in known_ids],
            rule_scope=constraint, item_level_target_coverage_proven=False,
            semantic_scope_review='agent review of archived source snapshot; not independent expert certification')
        cards.append(card)
        for split, designs in [('train',TRAIN_DESIGNS),('dev',DEV_DESIGNS)]:
            requests.append(dict(id=card['id']+'-'+split, unit=card['id'], split=split, category=category,
                designs=designs, sources=[{k:r[k] for k in ('id','title','scope','source_text','evidence','topics')} for r in selected],
                authoring_scope=constraint, allow_all_correct=True,
                answer_cardinalities={d:1+(card_index*len(designs)+j+(1 if split=='dev' else 0))%4 for j,d in enumerate(designs)},
                training_allowed=False, purpose='unreviewed_draft'))
    for i, category in RETENTION:
        r = rows[i]
        if groups[r['id']] not in dev_groups or r['id'] in known_ids:
            raise ValueError('retention source was trained in P1 or is outside frozen dev')
        requests.append(dict(id='expand-retain-'+str(i),unit='expand-retain-'+str(i),split='retention',category=category,
            designs=RETENTION_DESIGNS,sources=[{k:r[k] for k in ('id','title','scope','source_text','evidence','topics')}],
            authoring_scope='Prefer qualitative distinctions, causal limits, classification, and operational decisions. Do not replace domain reasoning with arithmetic drills. Preserve archived scope.',
            allow_all_correct=True,answer_cardinalities={d:1+(i+j)%4 for j,d in enumerate(RETENTION_DESIGNS)},
            training_allowed=False,purpose='unreviewed_draft'))
    selected_ids = {s for c in cards for s in c['source_ids']}
    catalog = [dict(id=r['id'],title=r['title'],topics=r['topics'],group=groups[r['id']],
        historical_split='dev' if groups[r['id']] in dev_groups else 'train',
        p1_domain=r['id'] in p1_ids,p1_replay=r['id'] in replay_ids,selected_for_expansion=r['id'] in selected_ids,
        evidence_keys=[e['key'] for e in r['evidence']],source_text_sha256=hashlib.sha256(r['source_text'].encode()).hexdigest()) for r in rows]
    # A hard allowlist protects the source-only generator from case-level audit fields.
    allowed = {'id','unit','split','category','designs','sources','authoring_scope','allow_all_correct','answer_cardinalities','training_allowed','purpose'}
    if any(set(r) != allowed for r in requests):
        raise ValueError('unexpected generator field')
    out.mkdir(parents=True, exist_ok=False)
    for name, content in [('catalog.private.jsonl',catalog),('cards.private.jsonl',cards),('requests.private.jsonl',requests)]:
        (out/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in content),encoding='utf-8',newline='\n')
    summary = dict(source_sha256=SOURCE_SHA,evidence_sha256=sha(evidence_path),replay_sha256=sha(old_path),
        evidence_check=evidence_check,source_units=205,source_groups=len(set(groups.values())),
        historical_dev_units=sum(groups[r['id']] in dev_groups for r in rows),historical_dev_groups=len(dev_groups),
        p1_domain_source_units=len(p1_ids),p1_replay_source_units=len(replay_ids),
        expansion_families=len(cards),expansion_source_units=len(selected_ids),
        source_units_absent_from_p1_training=len(selected_ids-known_ids),
        source_units_already_in_p1_training=len(selected_ids&known_ids),
        expansion_groups=len({groups[k] for k in selected_ids}),categories=dict(Counter(c['category'] for c in cards)),
        cards=cards,generation_requests=len(requests),requested_tasks=dict(Counter(d['split'] for d in requests for _ in d['designs'])),
        requested_train_answer_cardinalities=dict(Counter(n for r in requests if r['split']=='train' for n in r['answer_cardinalities'].values())),
        version='expansion_sources_v2',answer_cardinality_assignment_hidden_from_reviewer=True,
        retention_source_units=len(RETENTION),retention_source_groups=len({groups[rows[i]['id']] for i,_ in RETENTION}),
        retention_domains=dict(Counter(c for _,c in RETENTION)),
        retention_limitations=['Known historical development pool; not untouched final holdout.',
            'Warehouse/enterprise/Agent workflow retention still requires separately authored checks.',
            'No new retention items exist until generation and review; source counts are not task counts.'],
        p1_train_sha256=sha(resources/'llin-transfer-p1-20260916-03/train.private.jsonl'),
        source_only_generator=True,official_questions_in_generator=False,
        training_allowed=False,release_status='source_scoped_requests_only',
        prospective_training=dict(start='Step120 HF, fresh optimizer',retain_p1_unique_tasks=60,retain_legacy_replay=135,
            new_tasks_requested=126,domain_exposures_per_task=3,answer_only_per_task=1,explanation_per_task=2,
            maximum_records=693,maximum_steps_at_batch3=231,lr=5e-7,epochs=1,
            exact_token_budget='pending accepted task release; no truncation or filling rejected items'),
        files={p.name:sha(p) for p in out.glob('*.jsonl')})
    (out/'manifest.safe.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=prepare(a.resources,a.out)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cards','files')},ensure_ascii=False,indent=2))
