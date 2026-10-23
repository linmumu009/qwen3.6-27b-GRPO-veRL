// Run a copy beside a node_modules junction to the bundled spreadsheet runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const [inputDir, outputDir, qaDir] = process.argv.slice(2);
if (!inputDir || !outputDir || !qaDir) throw new Error('Expected source, output and QA directories');
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
const exported = [];
for (const [label, expectedCount] of [['LogistikaBench', 218], ['SC-bench知识题', 28]]) {
  const nativePath = path.join(inputDir, `${label}_原始基座.xlsx`);
  const bestPath = path.join(inputDir, `${label}_Best_Model.xlsx`);
  const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(nativePath));
  const best = await SpreadsheetFile.importXlsx(await FileBlob.load(bestPath));
  const qa = wb.worksheets.getItem('题目与回答');
  const bestQa = best.worksheets.getItem('题目与回答');
  const count = qa.getRange('B4').values[0][0];
  assert.equal(count, expectedCount);
  assert.equal(bestQa.getRange('B4').values[0][0], count);
  const lastRow = count + 7;
  const nativeRows = qa.getRange(`A8:K${lastRow}`).values;
  const bestRows = bestQa.getRange(`A8:K${lastRow}`).values;
  for (let r = 0; r < count; r++) {
    // Pair by stable hash and ID, and verify every shared source field.
    for (const c of [0, 1, 3, 6, 7, 8, 9]) {
      assert.deepEqual(nativeRows[r][c], bestRows[r][c], `Source mismatch at row ${r + 8}, column ${c}`);
    }
    assert.equal(nativeRows[r][5], '错误');
    assert.equal(bestRows[r][5], '正确');
  }
  const oldTable = qa.tables.items[0];
  assert.equal(oldTable.name, 'QuestionAnswers');
  oldTable.delete();
  // Move right to left so no original cell is overwritten before it is copied.
  for (let c = 10; c >= 3; c--) {
    const source = qa.getRangeByIndexes(0, c, lastRow, 1);
    const target = qa.getRangeByIndexes(0, c + 1, lastRow, 1);
    target.copyFrom(source, 'all');
    target.format.columnWidth = source.format.columnWidth;
  }
  const newAnswer = qa.getRange(`D7:D${lastRow}`);
  newAnswer.copyFrom(qa.getRange(`C7:C${lastRow}`), 'all');
  newAnswer.format.columnWidth = qa.getRange('C:C').format.columnWidth;
  qa.getRange(`D8:D${lastRow}`).values = bestRows.map(row => [row[2]]);
  qa.getRange('C7:D7').values = [['原始基座回答（完整选项正文）', 'Best Model回答（完整选项正文）']];
  qa.getRange('F7:G7').values = [['基座原始回答', '基座判定']];
  qa.getRange('L7').values = [['基座解析状态']];
  qa.getRange('A2').values = [[`${label} 原始基座与Best Model逐题对照`]];
  qa.getRange('B3').values = [[`Qwen3.6-27B 原始基座；${bestQa.getRange('B3').values[0][0]}`]];
  qa.getRange('B5').values = [['原始基座答错、最终 Best Model 答对的同一批题目，逐行对照。']];
  const table = qa.tables.add(`A7:L${lastRow}`, true, 'QuestionAnswers');
  table.style = 'TableStyleMedium2';
  table.showFilterButton = true;
  const options = wb.worksheets.getItem('完整选项');
  options.getRange('E1').values = [['基座是否选中']];
  wb.recalculate();
  const info = await wb.inspect({ kind: 'table', range: '题目与回答!A3:B4', include: 'values', tableMaxRows: 2, tableMaxCols: 2 });
  const errors = await wb.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!', options: { useRegex: true, maxResults: 10 }, maxChars: 1000 });
  await fs.writeFile(path.join(qaDir, `${label}.inspect.json`), JSON.stringify({ info: info.ndjson, errors: errors.ndjson }));
  for (const [sheetName, range, suffix] of [['题目与回答', 'A1:E10', 'questions'], ['完整选项', 'A1:F9', 'options']]) {
    const preview = await wb.render({ sheetName, range, scale: 1.2, format: 'png' });
    await fs.writeFile(path.join(qaDir, `${label}.${suffix}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  const target = path.join(outputDir, `${label}_基座与Best_Model对照.xlsx`);
  const xlsx = await SpreadsheetFile.exportXlsx(wb);
  await xlsx.save(target);
  try {
    await fs.rename(target + '.inspect.ndjson', path.join(qaDir, `${label}.xlsx.inspect.ndjson`));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  exported.push({ file: path.basename(target), questions: count, added_column: 'D', original_columns_preserved: 11 });
  console.log(JSON.stringify(exported.at(-1)));
}
await fs.writeFile(path.join(qaDir, 'export.safe.json'), JSON.stringify(exported, null, 2));
