'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const display = require('../web/numbers.js');
let checks = 0;
function equal(actual, expected) { assert.equal(actual, expected); checks++; }

// 경계값·음수·과학 표기에서 백분율 반올림과 그 외 수치의 절단을 구분하며 epsilon 없이 처리하는지 확인합니다.
for (const [input, expected] of [
  [0.9878999999999999,'98.8%'],[0.9878,'98.8%'],[0.58,'58.0%'],
  [0.29,'29.0%'],[-0.9878999999999999,'-98.8%'],[-0.000001,'0.0%'],
  [1,'100.0%'],[0,'0.0%'],[1e-7,'0.0%'],[null,'—'],[undefined,'—'],[NaN,'—'],[Infinity,'—'],[0.9875,'98.8%'],[0.9874999999999999,'98.7%'],[0.9994,'99.9%'],[0.9995,'100.0% (반올림)'],[0.999999,'100.0% (반올림)'],[-0.9995,'-100.0%'],
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
  setTimeout(){},AbortSignal,AbortController,URLSearchParams,fetch:async()=>({ok:true,text:async()=>JSON.stringify({total:0,trades:[]})}),console};
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
assert(elements['#validation-rows'].innerHTML.includes('98.8%')); checks++;
assert(!elements['#validation-rows'].innerHTML.includes('98.7%')); checks++;
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

// 아래 자료는 Node 검사 안에서만 쓰는 원장 fixture이며 실제 화면·비공개 원장에 저장하지 않습니다.
async function testTradeViews() {
  const requests = [];
  const views = {};
  function view() {
    let html = '';
    return {writes:0,textContent:'',value:'all',innerHTML:'',dataset:{},style:{},handlers:{},
      classList:{toggle(){},add(){},remove(){}},
      set innerHTML(value){html=value;this.writes++;},get innerHTML(){return html;},
      setAttribute(){},addEventListener(type,handler){this.handlers[type]=handler;},
      querySelector(){return view();},checkValidity(){return true;},scrollIntoView(){},focus(){}};
  }
  let result = {total:0,trades:[],as_of:'2026-11-10'};
  let responseStatus = 200;
  const browser = {DisplayNumbers:display,AbortController,AbortSignal,URLSearchParams,console,setTimeout(){},
    document:{querySelector(selector){return views[selector] ||= view();},querySelectorAll(){return [];},documentElement:view()},
    window:{matchMedia(){return {matches:false};}},localStorage:{getItem(){return null;},setItem(){}},
    fetch:async(url)=>{requests.push(url);return {ok:responseStatus===200,status:responseStatus,text:async()=>JSON.stringify(result)};}};
  vm.createContext(browser);
  vm.runInContext(source.replace(/poll\(\);\s*$/,''),browser);
  const fixture = {paper:{trade_count:0,positions:[{symbol:'TEST',name:'검사용 ETF',quantity:2.5,average_cost_usd:10.1999,cost_basis_usd:25.49975,cost_basis_krw:null,value_krw:40000,unrealized_pnl_krw:null,unrealized_pnl_pct:null}],
    orders:[{id:'검사 주문',symbol:'TEST',side:'매수',target_weight:0.25,status:'pending',reason:'새 가격 관측 대기'}],cash_movements:[]},
    validation:{metrics:[{model:'검사 모델',direction_accuracy:0.999999}]}};
  browser.fixture=fixture;
  vm.runInContext('render(fixture)',browser);
  await new Promise(resolve=>setImmediate(resolve));
  equal(requests.length,1);
  equal(requests[0],'/api/trades?offset=0&limit=50&side=all');
  assert(views['#position-rows'].innerHTML.includes('검사용 ETF'));checks++;
  assert(views['#position-rows'].innerHTML.includes('$10.19'));checks++;
  assert(views['#position-rows'].innerHTML.includes('>—<'));checks++;
  assert(views['#order-rows'].innerHTML.includes('체결 대기'));checks++;
  assert(views['#order-rows'].innerHTML.includes('25.0%'));checks++;
  assert(views['#validation-rows'].innerHTML.includes('100.0% (반올림)'));checks++;
  assert(views['#trade-rows'].innerHTML.includes('모의 체결이 없습니다'));checks++;
  const positionWrites=views['#position-rows'].writes, orderWrites=views['#order-rows'].writes;
  vm.runInContext('render(fixture)',browser);
  await new Promise(resolve=>setImmediate(resolve));
  equal(requests.length,1);
  equal(views['#position-rows'].writes,positionWrites);
  equal(views['#order-rows'].writes,orderWrites);
  result={total:125,trades:[{id:'검사 체결',symbol:'TEST',name:'검사용 ETF',side:'매수',executed_at:'2026-11-10T01:00:00Z',quantity:1,price_usd:10,amount_krw:14000,fee_krw:0,fx_rate:1400,reason:'검사 전용',realized_pnl_krw:null}],as_of:'2026-11-10'};
  fixture.paper.trade_count=125;
  vm.runInContext('render(fixture)',browser);
  await new Promise(resolve=>setImmediate(resolve));
  equal(requests.length,2);
  assert(views['#trade-rows'].innerHTML.includes('검사 전용'));checks++;
  assert(views['#trade-json'].textContent.includes('검사 체결'));checks++;
  views['#trade-next'].handlers.click();
  await new Promise(resolve=>setImmediate(resolve));
  equal(requests.at(-1),'/api/trades?offset=50&limit=50&side=all');
  views['#trade-side'].value='sell';
  views['#trade-side'].handlers.change();
  await new Promise(resolve=>setImmediate(resolve));
  equal(requests.at(-1),'/api/trades?offset=0&limit=50&side=sell');
  responseStatus=503;
  await vm.runInContext('fetchTrades()',browser);
  assert(views['#trade-status'].textContent.includes('준비 대기'));checks++;
  assert(views['#trade-rows'].innerHTML.includes('체결이 없다는 뜻은 아닙니다'));checks++;
  equal(views['#trade-retry'].disabled,false);
  responseStatus=200;
  result={total:0,trades:[]};
  views['#trade-retry'].handlers.click();
  await new Promise(resolve=>setImmediate(resolve));
  assert(views['#trade-rows'].innerHTML.includes('모의 체결이 없습니다'));checks++;
  equal(fixture.paper.positions[0].cost_basis_krw,null);
  assert(html.includes('aria-label="모의 체결 내역표'));checks++;
  assert(html.includes('ETF 체결 수에 포함하지 않습니다.'));checks++;
  assert(!html.includes('소수 둘째 자리 아래'));checks++;
  vm.runInContext('renderAccountDetails({})',browser);
  assert(views['#position-rows'].innerHTML.includes('응답을 기다리고'));checks++;
  assert(views['#order-rows'].innerHTML.includes('응답을 기다리고'));checks++;
  vm.runInContext('renderAccountDetails({positions:[],orders:[]})',browser);
  assert(views['#position-rows'].innerHTML.includes('현재 모의 보유 ETF가 없습니다'));checks++;
  assert(views['#order-rows'].innerHTML.includes('기록된 모의 주문이 없습니다'));checks++;
  vm.runInContext('renderAccountDetails({positions:[{name:"<img onerror=alert(1)>",symbol:"TEST"}],orders:[]})',browser);
  assert(!views['#position-rows'].innerHTML.includes('<img'));checks++;
  assert(views['#position-rows'].innerHTML.includes('&lt;img'));checks++;
  vm.runInContext('renderAccountDetails({cash_movements:[{rate:1356.3399,receipt:{applied_rate:1363.1216656}}]})',browser);
  assert(views['#cash-movement-rows'].innerHTML.includes('<td>1,356.3399</td><td>1,363.1216</td>'));checks++;
  vm.runInContext('renderAccountDetails({cash_movements:[{rate:1356.3399}]})',browser);
  assert(views['#cash-movement-rows'].innerHTML.includes('<td>1,356.3399</td><td>—</td>'));checks++;
  console.log(`화면 표시 검사 ${checks}개 통과 · 반올림·원값·보유·주문·거래 페이지·조회 실패·UTF-8`);
}
testTradeViews().catch(error=>{console.error(error);process.exitCode=1;});

