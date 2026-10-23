// Run a copy in a task directory whose node_modules points to the bundled runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const [inputPath, outputDir, qaDir] = process.argv.slice(2);
if (!inputPath || !outputDir || !qaDir) throw new Error('Expected pairs.json, output directory, QA directory');
const payload = JSON.parse(await fs.readFile(inputPath, 'utf8'));
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
const specs = [
  ['LogistikaBench', 'LogistikaBench'],
  ['SC-bench-knowledge', 'SC-bench知识题'],
];
const headers = ['配对编号', '题目', '模型回答（完整选项正文）', '标准答案（完整选项正文）',
  '模型原始回答', '判定', '原题编号', '题目指纹', '题型', '分类', '解析状态'];
const widths = [16, 65, 65, 65, 36, 12, 18, 66, 20, 24, 14];
function literal(value) {
  if (typeof value !== 'string') return value;
  assert(value.length <= 32767, 'Excel cell text exceeds 32767 characters');
  assert(!/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(value), 'Invalid XML control in source');
  return value.startsWith('=') ? "'" + value : value;
}
function write(sheet, row, matrix) {
  sheet.getRangeByIndexes(row, 0, matrix.length, matrix[0].length).values = matrix.map(r => r.map(literal));
}
function lines(text, width) {
  return String(text ?? '').split('\n').reduce((n, s) => {
    let w = 0;
    for (const c of s) w += /[\u2e80-\uffef]/u.test(c) ? 2 : 1;
    return n + Math.max(1, Math.ceil(w / (width - 3)));
  }, 0);
}
function baseStyle(sheet, end, cols, colWidths) {
  sheet.showGridLines = false;
  const range = sheet.getRangeByIndexes(0, 0, end, cols);
  range.format.font = { name: 'Arial', size: 10, color: '#202D40' };
  range.format.verticalAlignment = 'top';
  range.format.wrapText = true;
  range.format.numberFormat = '@';
  for (let c = 0; c < cols; c++) {
    sheet.getRangeByIndexes(0, c, end, 1).format.columnWidth = colWidths[c];
  }
}
function heading(sheet, row, count) {
  const range = sheet.getRangeByIndexes(row, 0, 1, count);
  range.format.fill = '#203B5D';
  range.format.font = { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' };
  range.format.horizontalAlignment = 'center';
  range.format.verticalAlignment = 'center';
  range.format.rowHeight = 32;
}
const exported = [];
for (const [dataset, displayName] of specs) {
  const pairs = payload.pairs.filter(p => p.dataset === dataset);
  assert.equal(pairs.length, payload.manifest.by_dataset[dataset]);
  for (const role of ['native', 'best']) {
    const roleName = role === 'native' ? '原始基座' : 'Best_Model';
    const wb = Workbook.create();
    const qa = wb.worksheets.add('题目与回答');
    const opts = wb.worksheets.add('完整选项');
    const rows = pairs.map(p => {
      const response = p[role];
      assert.equal(response.correct, role === 'best');
      const resolved = response.parsed.map(i => `[${i}] ${p.options[i]}`).join('\n');
      assert.deepEqual(response.answer_texts, response.parsed.map(i => p.options[i]));
      return [p.pair_id, p.question,
        response.parse_ok ? (resolved || '未选择任何选项') : '解析失败，请参见模型原始回答',
        p.expected.map(i => `[${i}] ${p.options[i]}`).join('\n'),
        response.prediction, response.correct ? '正确' : '错误',
        p.source_id, p.item_hash, p.question_type, p.category,
        response.parse_ok ? '成功' : '失败'];
    });
    assert(pairs.every(p => !p.native.reasoning && !p.best.reasoning), 'Unexpected reasoning needs an explicit output column');
    baseStyle(qa, rows.length + 7, headers.length, widths);
    write(qa, 1, [[`${displayName} ${roleName}题目与回答`]]);
    qa.getRange('A2').format.font = { name: 'Arial', size: 15, bold: true, color: '#203B5D' };
    qa.getRange('A2:K2').format.rowHeight = 28;
    qa.getRange('A2').format.wrapText = false;
    write(qa, 2, [['模型', role === 'native' ? 'Qwen3.6-27B 原始基座' : '最终 Best Model（追加训练第150步，累计第253步）']]);
    write(qa, 3, [['筛选题数', pairs.length]]);
    qa.getRange('B4').setNumberFormat('#,##0');
    write(qa, 4, [['筛选条件', '原始基座答错，最终 Best Model 答对。两份文件按同一配对编号排列。']]);
    write(qa, 5, [['选项说明', '全部选项见“完整选项”。方括号内为原评测从0开始的序号。']]);
    qa.getRange('A3:K6').format.rowHeight = 30;
    qa.getRange('B3:B6').format.font = { name: 'Arial', size: 10, color: '#43556A' };
    write(qa, 6, [headers]);
    write(qa, 7, rows);
    const table = qa.tables.add(`A7:K${rows.length + 7}`, true, 'QuestionAnswers');
    table.showFilterButton = true;
    heading(qa, 6, headers.length);
    for (let i = 0; i < rows.length; i++) {
      const height = Math.max(34, ...rows[i].map((v, c) => lines(v, widths[c]) * 14 + 10));
      assert(height <= 409, `Question row too tall: ${rows[i][0]}`);
      qa.getRangeByIndexes(i + 7, 0, 1, headers.length).format.rowHeight = height;
    }
    qa.freezePanes.freezeRows(7);
    qa.freezePanes.freezeColumns(1);
    qa.tabColor = '#203B5D';

    const optionRows = pairs.flatMap(p => p.options.map((text, i) => [
      p.pair_id, i, text, p.expected.includes(i) ? '是' : '否',
      p[role].parse_ok && p[role].parsed.includes(i) ? '是' : '否', p.source_id,
    ]));
    const optionWidths = [16, 18, 100, 16, 16, 18];
    baseStyle(opts, optionRows.length + 1, 6, optionWidths);
    write(opts, 0, [['配对编号', '选项序号（从0开始）', '完整选项正文', '是否标准答案', '模型是否选中', '原题编号']]);
    write(opts, 1, optionRows);
    const ot = opts.tables.add(`A1:F${optionRows.length + 1}`, true, 'CompleteOptions');
    ot.showFilterButton = true;
    heading(opts, 0, 6);
    for (let i = 0; i < optionRows.length; i++) {
      opts.getRangeByIndexes(i + 1, 0, 1, 6).format.rowHeight = Math.max(28,
        ...optionRows[i].map((value, c) => lines(value, optionWidths[c]) * 14 + 8));
    }
    opts.getRange(`B2:B${optionRows.length + 1}`).setNumberFormat('0');
    opts.freezePanes.freezeRows(1);
    opts.freezePanes.freezeColumns(1);
    wb.recalculate();
    const info = await wb.inspect({ kind: 'table', range: '题目与回答!A3:B4', include: 'values', tableMaxRows: 2, tableMaxCols: 2 });
    const errors = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 10 }, maxChars: 1000 });
    const stem = `${displayName}_${roleName}`;
    await fs.writeFile(path.join(qaDir, `${stem}.inspect.json`), JSON.stringify({ info: info.ndjson, errors: errors.ndjson }));
    for (const [sheetName, range, suffix] of [['题目与回答', 'A1:F10', 'questions'], ['完整选项', 'A1:F9', 'options']]) {
      const image = await wb.render({ sheetName, range, scale: 1.3, format: 'png' });
      await fs.writeFile(path.join(qaDir, `${stem}.${suffix}.png`), new Uint8Array(await image.arrayBuffer()));
    }
    const target = path.join(outputDir, `${stem}.xlsx`);
    const xlsx = await SpreadsheetFile.exportXlsx(wb);
    await xlsx.save(target);
    try {
      await fs.rename(target + '.inspect.ndjson', path.join(qaDir, `${stem}.xlsx.inspect.ndjson`));
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    exported.push({ file: path.basename(target), dataset, role, questions: rows.length, option_rows: optionRows.length });
    console.log(JSON.stringify(exported.at(-1)));
  }
}
assert.equal(exported.length, 4);
await fs.writeFile(path.join(qaDir, 'export.safe.json'), JSON.stringify(exported, null, 2));
