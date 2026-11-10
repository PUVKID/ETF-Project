'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const display = require('../web/numbers.js');
let checks = 0;
function equal(actual, expected) { assert.equal(actual, expected); checks++; }

// 경계값·음수·과학 표기에서 반올림이나 epsilon 없이 자르는지 확인합니다.
for (const [input, expected] of [
  [0.9878999999999999,'98.78%'],[0.9878,'98.78%'],[0.58,'58.00%'],
  [0.29,'29.00%'],[-0.9878999999999999,'-98.78%'],[-0.000001,'0.00%'],
  [1,'100.00%'],[0,'0.00%'],[1e-7,'0.00%'],[null,'—'],[undefined,'—'],[NaN,'—'],[Infinity,'—'],
]) equal(display.percent(input), expected);
for (const [input, places, expected] of [
  [98.7899,2,'98.78'],[-98.7899,2,'-98.78'],[1.9999999999999998,2,'1.99'],
  [1234.999,2,'1,234.99'],[1.005,2,'1.00'],[-0.009,2,'0.00'],
  [0.000001999999,6,'0.000001'],[1.23456789e-7,10,'0.0000001234'],
  [1.23456789e21,2,'1,234,567,890,000,000,000,000.00'],
]) equal(display.number(input, places), expected);
equal(display.truncateDecimal('0.987899999999999999999999',2,2),'98.78');
equal(display.truncateDecimal('-98.789999999999999999',2),'-98.78');
equal(display.truncateDecimal('1e100000000',2),'—');
equal(display.number('98.789',2),'—');

const root = path.resolve(__dirname, '..');
const source = fs.readFileSync(path.join(root,'web/app.js'),'utf8');
const html = fs.readFileSync(path.join(root,'web/index.html'),'utf8');
const elements = {};
function element() {
  return {textContent:'',innerHTML:'',dataset:{},style:{},open:false,
    classList:{toggle(){},add(){},remove(){}},setAttribute(){},addEventListener(){},
    querySelector(){return element();},checkValidity(){return true;},scrollIntoView(){},focus(){}};
}
const context = {DisplayNumbers:display,
  document:{querySelector(selector){return elements[selector] ||= element();},querySelectorAll(){return [];},documentElement:element()},
  window:{matchMedia(){return {matches:false};}},localStorage:{getItem(){return null;},setItem(){}},
  setTimeout(){},AbortSignal,console};
vm.createContext(context);
vm.runInContext(source.replace(/poll\(\);\s*$/,''),context);
const state = {training:{loss:0.12345678901234567,learning_rate:1.23456789e-7},
  validation:{metrics:[{model:'검사 모델',split:'test',mae:0.12345678901234567,mae_krw:170.987654,mape:0.9878999999999999,direction_accuracy:0.58}]},
  simulation:{strategies:[{name:'동일가중 단순 보유',total_return:0.9878999999999999}]}};
context.fixture = state;
const before = JSON.stringify(state);
vm.runInContext('render(fixture)',context);
equal(state.training.loss,0.12345678901234567);
equal(state.validation.metrics[0].mape,0.9878999999999999);
assert(elements['#validation-rows'].innerHTML.includes('98.78%')); checks++;
assert(!elements['#validation-rows'].innerHTML.includes('98.79%')); checks++;
equal(vm.runInContext('maeText({mae_krw:170.987654})',context),'170.98원');
vm.runInContext('latestRawResponse = \'{"ratio":0.123456789012345678901234567890}\'; latestRawSource="/api/state"; takeRawSnapshot()',context);
equal(elements['#raw-json'].textContent,'{"ratio":0.123456789012345678901234567890}');
assert(elements['#raw-value-rows'].innerHTML.includes('0.9878999999999999')); checks++;
assert(elements['#raw-value-rows'].innerHTML.includes('0.12345678901234566') || elements['#raw-value-rows'].innerHTML.includes(String(state.training.loss))); checks++;
assert(elements['#raw-value-rows'].innerHTML.includes('1.23456789e-7')); checks++;
const fixedSnapshot = elements['#raw-json'].textContent;
vm.runInContext('latestRawResponse = \'{"ratio":0.999999}\'; render({training:{loss:1}})',context);
equal(elements['#raw-json'].textContent,fixedSnapshot);
vm.runInContext('takeRawSnapshot()',context);
equal(elements['#raw-json'].textContent,'{"ratio":0.999999}');
assert(elements['#raw-value-rows'].innerHTML.includes('>—<')); checks++;
// 표시 과정에서 입력 fixture의 원래 수치가 바뀌지 않았는지 확인합니다.
const after = JSON.parse(JSON.stringify(state));
const original = JSON.parse(before);
equal(after.training.loss,original.training.loss);
equal(after.validation.metrics[0].mae,original.validation.metrics[0].mae);

assert(html.includes('tabindex="0" role="region" aria-label="예측 검증표')); checks++;
assert(html.indexOf('src="/numbers.js"') < html.indexOf('src="/app.js"')); checks++;
for (const file of ['web/index.html','web/style.css','web/app.js','web/numbers.js']) {
  const text = new TextDecoder('utf-8',{fatal:true}).decode(fs.readFileSync(path.join(root,file)));
  equal(text.includes('\uFFFD'),false);
}
console.log(`화면 표시 검사 ${checks}개 통과 · 숫자 원값·버림·원문 고정·접근성 구조·UTF-8`);
