// Use the bundled runtime through a task-local node_modules junction.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const [input, outDir, qaDir] = process.argv.slice(2);
if (!input || !outDir || !qaDir) throw new Error('Expected private pairs, output directory, QA directory');
const source = JSON.parse(await fs.readFile(input, 'utf8'));
assert.equal(source.pending_aime.length, 0, 'Resolve pending model identity before using this exporter');
await fs.mkdir(outDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
const priority = ['AIME24', 'AIME25', 'GSM8K_test', 'MGSM-zh', 'CMATH'];
const order = name => priority.includes(name) ? priority.indexOf(name) : 100;
const cases = [...source.cases].sort((a, b) => order(a.dataset) - order(b.dataset)
  || a.dataset.localeCompare(b.dataset) || a.task_id.localeCompare(b.task_id));
function userQuestion(raw) {
  try {
    const messages = JSON.parse(raw);
    const users = messages.filter(m => m.role === 'user');
    return users.at(-1).content;
  } catch { return raw; }
}
cases.forEach((c, i) => {
  c.case_id = `OS-${String(i + 1).padStart(4, '0')}`;
  c.user_question = userQuestion(c.question);
  assert.equal(typeof c.user_question, 'string');
  assert.equal(c.native.scoring.correct, false);
  assert.equal(c.trained.scoring.correct, true);
  assert.notEqual(c.trained.prediction.finish_reason, 'length');
  assert.equal(c.native.prediction.prompt, c.trained.prediction.prompt);
  assert.equal(c.native.scoring.golden_answer, c.trained.scoring.golden_answer);
});
const wb = Workbook.create();
const overview = wb.worksheets.add('评测汇总');
const main = wb.worksheets.add('改善题对照');
const detail = wb.worksheets.add('完整作答');
function value(v) {
  if (typeof v !== 'string') return v;
  // Excel XML cannot store these controls. Display their escape notation;
  // the accompanying JSON keeps every original character unchanged.
  // Include CR: literal carriage returns are normalized by XML readers.
  const text = v.replace(/[\x00-\x08\x0b-\x1f]/g,
    ch => '\\u' + ch.charCodeAt(0).toString(16).padStart(4, '0'));
  assert(text.length <= 32767);
  return text.startsWith('=') ? "'" + text : text;
}
function write(sheet, row, matrix) {
  sheet.getRangeByIndexes(row, 0, matrix.length, matrix[0].length).values = matrix.map(r => r.map(value));
}
function lineCount(text, width) {
  return String(value(text ?? '')).split('\n').reduce((total, line) => {
    let units = 0;
    for (const ch of line) units += /[\u2e80-\uffef]/u.test(ch) ? 2 : 1;
    return total + Math.max(1, Math.ceil(units / (width - 3)));
  }, 0);
}
function preview(text, width) {
  const raw = String(text ?? '');
  // Presentation excerpts only. Exact unmodified strings are on 完整作答 and in JSON.
  let displayed = raw;
  let chars = Array.from(raw);
  while (displayed.length > 700 || lineCount(displayed, width) > 12) {
    chars = chars.slice(0, Math.max(1, chars.length - 50));
    displayed = chars.join('');
  }
  if (displayed.length !== raw.length) displayed += '\n…（全文见“完整作答”）';
  return displayed || '（空）';
}
function style(sheet, rowCount, widths) {
  sheet.showGridLines = false;
  const area = sheet.getRangeByIndexes(0, 0, rowCount, widths.length);
  area.format.font = { name: 'Arial', size: 10, color: '#203047' };
  area.format.numberFormat = '@';
  area.format.wrapText = true;
  area.format.verticalAlignment = 'top';
  widths.forEach((width, c) => { sheet.getRangeByIndexes(0, c, rowCount, 1).format.columnWidth = width; });
}
function table(sheet, row, headers, rows, widths, name) {
  style(sheet, row + rows.length + 1, widths);
  write(sheet, row, [headers]);
  if (rows.length) write(sheet, row + 1, rows);
  const end = String.fromCharCode(64 + headers.length);
  const t = sheet.tables.add(`A${row + 1}:${end}${row + rows.length + 1}`, true, name);
  t.style = 'TableStyleMedium2';
  t.showFilterButton = true;
  const header = sheet.getRangeByIndexes(row, 0, 1, headers.length);
  header.format.fill = '#203B5D';
  header.format.font = { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' };
  header.format.rowHeight = 32;
  header.format.horizontalAlignment = 'center';
  header.format.verticalAlignment = 'center';
  rows.forEach((r, i) => {
    const height = Math.max(30, ...r.map((v, c) => lineCount(v, widths[c]) * 14 + 10));
    assert(height <= 409, `Row exceeds Excel display height: ${name} ${i}`);
    sheet.getRangeByIndexes(row + 1 + i, 0, 1, headers.length).format.rowHeight = height;
  });
  sheet.freezePanes.freezeRows(row + 1);
  sheet.freezePanes.freezeColumns(2);
}
const status = {
  included: '纳入', included_user_confirmed_identity: '用户已确认模型归属',
  excluded_prompt_mismatch: '题面提示不一致，未纳入',
  excluded_model_tag_conflict: '模型快照标签冲突，未纳入',
  no_complete_improvement: '无符合条件的完整案例',
};
const audits = [...source.manifest.audits].sort((a, b) => order(a.dataset) - order(b.dataset)
  || a.dataset.localeCompare(b.dataset) || b.mode.localeCompare(a.mode));
const summaryRows = audits.map(a => [a.dataset, a.mode, a.items, a.native_correct, a.trained_correct,
  a.recorded_gains, a.exported_items, a.recorded_losses, status[a.status] || a.status]);
table(overview, 9, ['数据集', '推理模式', '全量题数', '基座答对', '续训答对', '历史错转对', '纳入案例', '历史对转错', '核验说明'], summaryRows,
  [24, 22, 14, 14, 14, 16, 14, 16, 42], 'EvaluationOverview');
overview.getRange(`C11:H${summaryRows.length + 10}`).setNumberFormat('#,##0');
write(overview, 1, [['开源评测：原始基座与Step120改善题对照']]);
overview.getRange('A2').format.font = { name: 'Arial', size: 15, bold: true, color: '#203B5D' };
overview.getRange('A2').format.wrapText = false;
write(overview, 3, [['纳入案例', cases.length, '数据集数', Object.keys(source.manifest.by_dataset).length]]);
overview.getRange('B4').setNumberFormat('#,##0');
overview.getRange('D4').setNumberFormat('#,##0');
const notes = [
  '模型：Qwen3.6-27B原始基座，对照开源续训Step120。所用结果来自人均历史评测。',
  '筛选：同题面、同标准答案，基座评分错误、续训评分正确。接口报错及续训截断回答已排除。',
  '来源：AIME的30K模型身份已由用户确认，原快照标签冲突保留记录。GSM8K按历史标签配对，快照权重路径为占位符。',
  '阅读：长内容全文见“完整作答”。极少数控制字符以转义形式展示，附带JSON保留原文。',
  '口径：沿用当时逐题判分，未重新推理。错转对案例数不等于全量净增正确数，也不证明差异只由训练导致。',
];
notes.forEach((note, i) => {
  write(overview, i + 4, [[note]]);
  overview.getRangeByIndexes(i + 4, 0, 1, 9).format.rowHeight = 27;
  overview.getRangeByIndexes(i + 4, 0, 1, 1).format.wrapText = false;
});
const widths = [15, 23, 70, 55, 55, 45, 12, 12, 19, 19, 25, 45, 22];
const mainRows = cases.map(c => [c.case_id, c.dataset, preview(c.user_question, 70),
  preview(c.native.scoring.extracted_answer, 55), preview(c.trained.scoring.extracted_answer, 55),
  preview(c.gold_answer, 45), '错误', '正确', c.native.prediction.finish_reason,
  c.trained.prediction.finish_reason, c.task_id, preview(c.source_status, 45), c.mode]);
table(main, 0, ['配对编号', '数据集', '题目（长文全文见附表）', '原始基座答案', '开源续训Step120答案',
  '标准答案或约束', '基座判定', '续训判定', '基座结束状态', '续训结束状态', '原题编号', '来源说明', '推理模式'],
  mainRows, widths, 'ImprovedQuestionPairs');

const details = [];
for (const c of cases) {
  const fields = [
    ['题目（用户消息全文）', c.user_question], ['原始请求消息JSON', c.question],
    ['标准答案或约束', c.gold_answer],
    ['基座原始回答', c.native.prediction.response],
    ['续训原始回答', c.trained.prediction.response],
    ['基座独立推理字段', c.native.prediction.reasoning_content || ''],
    ['续训独立推理字段', c.trained.prediction.reasoning_content || ''],
    ['基座提取答案', c.native.scoring.extracted_answer],
    ['续训提取答案', c.trained.scoring.extracted_answer],
    ['基座判分依据', c.native.scoring.evidence || ''],
    ['续训判分依据', c.trained.scoring.evidence || ''],
  ];
  for (const [field, text] of fields) {
    const chars = Array.from(String(text ?? ''));
    const chunks = [];
    let chunk = '';
    for (const ch of chars) {
      if (chunk.length >= 650 || lineCount(chunk + ch, 110) > 18) {
        chunks.push(chunk); chunk = '';
      }
      chunk += ch;
    }
    chunks.push(chunk);
    chunks.forEach((part, i) => details.push([c.case_id, c.dataset, field, i + 1, part]));
  }
}
table(detail, 0, ['配对编号', '数据集', '内容类型', '分段序号', '完整内容（按序拼接，无删减）'],
  details, [15, 23, 27, 15, 110], 'CompleteResponses');
detail.getRange(`D2:D${details.length + 1}`).setNumberFormat('0');
overview.tabColor = '#203B5D'; main.tabColor = '#426993';
wb.recalculate();
const info = await wb.inspect({ kind: 'table', range: '评测汇总!A4:D4', include: 'values', tableMaxRows: 1, tableMaxCols: 4 });
const errors = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 10 }, maxChars: 1500 });
await fs.writeFile(path.join(qaDir, 'inspect.json'), JSON.stringify({ info: info.ndjson, errors: errors.ndjson }));
for (const [sheetName, range, name] of [['评测汇总', 'A1:I17', 'overview'], ['改善题对照', 'A1:F6', 'comparison'], ['完整作答', 'A1:E8', 'complete']]) {
  const png = await wb.render({ sheetName, range, scale: 1.25, format: 'png' });
  await fs.writeFile(path.join(qaDir, name + '.png'), new Uint8Array(await png.arrayBuffer()));
}
const stem = '开源评测_基座与Step120改善题对照';
const target = path.join(outDir, stem + '.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(target);
try { await fs.rename(target + '.inspect.ndjson', path.join(qaDir, 'workbook.inspect.ndjson')); }
catch (e) { if (e.code !== 'ENOENT') throw e; }
await fs.writeFile(path.join(outDir, stem + '.json'), JSON.stringify({
  metadata: source.manifest, cases,
}, null, 2) + '\n');
const exported = { questions: cases.length, datasets: Object.keys(source.manifest.by_dataset).length,
  detail_rows: details.length, summary_rows: summaryRows.length, workbook: path.basename(target) };
await fs.writeFile(path.join(qaDir, 'export.safe.json'), JSON.stringify(exported, null, 2));
console.log(JSON.stringify(exported));
