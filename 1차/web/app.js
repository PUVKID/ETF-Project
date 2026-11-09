'use strict';
const $ = (selector) => document.querySelector(selector);
const safe = (value) => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const number = (value, digits = 2) => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString('ko-KR', {maximumFractionDigits:digits, minimumFractionDigits:digits}) : '—';
const percent = (value) => typeof value === 'number' && Number.isFinite(value) ? `${number(value * 100)}%` : '—';
const money = (value) => typeof value === 'number' && Number.isFinite(value) ? `$${number(value)}` : '—';
const won = (value) => typeof value === 'number' && Number.isFinite(value) ? `${number(value,0)}원` : '—';
const hasKrwMae = (metric) => typeof metric.mae_krw === 'number' && Number.isFinite(metric.mae_krw);
const maeText = (metric) => hasKrwMae(metric) ? `${number(metric.mae_krw,2)}원` : typeof metric.mae === 'number' && Number.isFinite(metric.mae) ? `${number(metric.mae,4)} USD` : '—';
const dateText = (value) => { if (!value) return '—'; const date = new Date(value); return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('ko-KR', {month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}); };
const statusText = (status) => ({idle:'대기',waiting:'다음 회차 대기',backoff:'재시도 대기',running:'진행 중',collecting:'수집 중',loading:'불러오는 중',downloading:'모델 내려받는 중',training:'학습 중',evaluating:'검증 기록 중',validating:'검증 중',simulating:'모의운용 중',completed:'완료',complete:'완료',success:'완료',ready:'준비됨',failed:'오류',error:'오류',stopped:'중지됨',stopping:'중지 중',pending:'대기',not_started:'시작 전',unavailable:'사용 불가',skipped:'건너뜀'}[status] || status || '대기');
let latestState = null;
let posting = false;
let paused = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let hasScenario = false;

// 보유 내역과 수치는 로컬 API에서만 읽으며 화면 오류도 원문으로 표시합니다.
function normalize(raw) {
  const state = raw && typeof raw === 'object' ? raw : {};
  for (const key of ['paper','data','training','validation','simulation','job']) if (!state[key] || typeof state[key] !== 'object') state[key] = {};
  for (const key of ['history','logs','notes']) if (!Array.isArray(state[key])) state[key] = [];
  return state;
}

function render(state) {
  latestState = normalize(state);
  renderPaper(latestState.paper);
  const {data, training, validation, simulation, job, history, logs, notes} = latestState;
  $('#clock').textContent = `상태 확인 ${dateText(new Date().toISOString())}`;
  $('#data-time').textContent = `시세 기준 ${dateText(data.as_of)} · 모의계좌 평가는 원화 기준`;
  drawHistory(history);
  const strategies = Array.isArray(simulation.strategies) ? simulation.strategies : [];
  const strategy = strategies.find(item => item.name === '동일가중 단순 보유') || {};
  $('#performance-metrics').innerHTML = [['동일가중 보유 · 누적수익률',percent(strategy.total_return)],['최대 낙폭',percent(strategy.max_drawdown)],['동일가중 보유 · 연복리 수익률',percent(strategy.cagr)],['회전율',percent(strategy.turnover)]].map(([label,value]) => `<div><span>${label}</span><strong>${safe(value)}</strong></div>`).join('');
  $('#job-status').textContent = `${job.continuous ? '반복 운용 · ' : ''}${statusText(job.status)}`;
  $('#job-message').textContent = job.message || '현재 실행 중인 작업이 없습니다.';
  const progress = typeof training.step === 'number' && training.total_steps > 0 ? Math.max(0,Math.min(100,training.step / training.total_steps * 100)) : 0;
  $('#training-progress').style.width = `${progress}%`;
  $('#learning-stats').innerHTML = [['학습 회차',`${number(training.round,0)}회`],['이번 회차 단계',`${number(training.step,0)} / ${number(training.total_steps,0)}`],['누적 학습 단계',number(training.cumulative_steps,0)],['마지막 저장',dateText(training.checkpoint_saved_at)],['새로 수집한 일봉',`${number(data.new_rows,0)}개`],['최근 손실',number(training.loss,6)]].map(([label,value])=>`<div><dt>${label}</dt><dd>${safe(value)}</dd></div>`).join('');
  const resumeState = training.resumed === true ? '이전 latest 체크포인트에서 이어 학습' : training.resumed === false ? '사전학습 원본에서 첫 학습' : '이어 학습 상태 확인 중';
  $('#checkpoint-status').textContent = `Chronos-2 · ${resumeState}${training.checkpoint ? ` · 저장본 ${training.checkpoint}` : ''}`;
  $('#next-run').textContent = `최근 학습 종료 ${dateText(training.finished_at)}${job.next_run_at ? ` · 다음 확인 ${dateText(job.next_run_at)}` : ''}${training.new_rows === 0 && training.round > 0 ? ' · 같은 일봉 재사용 회차' : ''}`;
  $('#data-source').textContent = `${data.source || '출처 미확인'} · ${data.start || '—'} ~ ${data.end || '—'} · ${number(data.rows,0)}행 · ${data.adjustment || '조정 여부 미확인'}`;
  $('#validation-status').textContent = statusText(validation.status);
  const metrics = Array.isArray(validation.metrics) ? validation.metrics : [];
  const hasKrw = metrics.some(hasKrwMae);
  $('#mae-heading').textContent = metrics.length ? hasKrw ? metrics.every(hasKrwMae) ? 'MAE(원 환산)' : 'MAE(원 환산 / USD)' : 'MAE(USD)' : 'MAE(통화 표시)';
  $('#validation-rows').innerHTML = metrics.length ? metrics.map(m => `<tr><td>${safe(m.model)}</td><td>${safe(m.split)}</td><td>${safe(maeText(m))}</td><td>${safe(percent(m.mape))}</td><td>${safe(percent(m.direction_accuracy))}</td><td>${safe(number(m.n,0))}</td></tr>`).join('') : '<tr><td colspan="6" class="empty">검증 결과가 아직 없습니다.</td></tr>';
  const assumptions = Array.isArray(simulation.assumptions) ? simulation.assumptions : [];
  $('#validation-notes').innerHTML = [...notes,...assumptions].map(note => `<p>${safe(typeof note === 'string' ? note : JSON.stringify(note))}</p>`).join('');
  $('#logs').innerHTML = logs.length ? logs.slice(-20).reverse().map(log => `<div class="log-row ${/error|failed/i.test(log.level || '') ? 'error' : ''}"><time>${safe(dateText(log.time))}</time><span>${safe(log.level || '정보')}</span><span>${safe(log.message)}</span></div>`).join('') : '<p class="empty">작업 기록이 아직 없습니다.</p>';
  renderStages({data,training,validation,simulation,job});
  if (hasScenario) calculateScenario();
}

// 모의계좌는 실제 보유와 분리하고 출처·기준일이 있는 환율만 환산에 씁니다.
function renderPaper(paper) {
  const fx = paper.fx || {};
  const hasFx = typeof fx.rate === 'number' && Number.isFinite(fx.rate) && fx.rate > 0 && Boolean(fx.source) && Boolean(fx.as_of);
  $('#paper-equity').innerHTML = `${safe(hasFx ? number(paper.current_balance_krw,0) : '—')}<span> KRW</span>`;
  $('#paper-status').textContent = paper.message || (hasFx ? `모의계좌 기준 ${dateText(paper.as_of)}` : '실제 환율 확인 후 가상계좌를 집계합니다.');
  $('#paper-cash-krw').textContent = won(paper.cash_krw);
  $('#paper-cash-usd').innerHTML = `${safe(hasFx ? won(paper.cash_usd_value_krw) : '환산 대기')}<small>${safe(money(paper.cash_usd))} USD</small>`;
  $('#paper-fees').textContent = hasFx ? won(paper.fees_krw) : '환산 대기';
  $('#paper-fx-fees').textContent = won(paper.exchange_cost_krw);
  const positions = Array.isArray(paper.positions) ? paper.positions : [];
  const valuesReady = positions.every(p => typeof p.value_krw === 'number' && Number.isFinite(p.value_krw));
  $('#paper-position-value').textContent = hasFx && valuesReady ? won(positions.reduce((sum,p)=>sum+p.value_krw,0)) : '—';
  $('#paper-pnl').textContent = hasFx ? `${won(paper.pnl_krw)} · ${percent(paper.pnl_pct)}` : '환산 대기';
  $('#paper-positions').innerHTML = positions.length ? positions.map(p=>`<div class="holding-row"><strong>${safe(p.symbol)}</strong><span>${safe(number(p.quantity,4))}주</span><span>${safe(hasFx ? won(p.value_krw) : '환산 대기')}<small>주당 ${safe(hasFx ? won(p.price_krw) : '—')} · ${safe(money(p.price_usd))}</small></span></div>`).join('') : `<p class="empty">${hasFx ? '아직 가상 보유가 없습니다. 다음 거래 관측을 기다립니다.' : '환율 확인 전 · 가상 매수 내역이 없습니다.'}</p>`;
  const receipt = paper.fx_receipt || {};
  $('#paper-fx').textContent = hasFx ? `시장환율 1 USD = ${won(fx.rate)} · ${fx.source} · 기준 ${dateText(fx.as_of)} · 수집 ${dateText(fx.fetched_at)}${typeof receipt.applied_rate === 'number' ? ` · 최초 모의 적용환율 ${number(receipt.applied_rate,4)}원/USD · 기록된 스프레드 ${number(receipt.spread_bps,0)}bp` : ''} · 토스 실제 적용환율이 아닌 모의 가정입니다. 원화 평가에 환율 변동을 반영하며 양도소득세·배당세는 미반영합니다.` : 'KRW ↔ USD 환산 대기 · 환율 출처와 시각 확인 전입니다. 시장환율은 토스 실제 적용환율이 아닙니다.';
  const policy = paper.fee_policy || {};
  $('#paper-fee-policy-status').textContent = `${policy.account_confirmation || '기본 0.1% 적용은 사용자가 확인했습니다. 계좌 API는 연결하지 않았습니다.'}${policy.checked_at ? ` · 서버 정책 확인일 ${policy.checked_at}` : ' · 서버 비용정책 정보 확인 대기'}. 실제 은행 호가를 연결하지 않아 토스의 실제 체결·환전 비용과 완전히 같지 않습니다.`;
}

function renderStages(state) {
  const action = state.job.action;
  const aliases = {data:['refresh','collect','collecting'],training:['train','training'],validation:['validate','validating'],simulation:['simulate','simulating']};
  for (const [key, actions] of Object.entries(aliases)) {
    const element = $(`[data-stage="${key}"]`);
    const status = state[key].status;
    const active = /running|training|collecting|validating|simulating|loading|downloading|evaluating/.test(status || '') || (state.job.status === 'running' && actions.includes(action));
    element.classList.toggle('active', active);
    element.classList.toggle('complete', /complete|ready|success/.test(status || ''));
    element.classList.toggle('failed', /error|failed/.test(status || ''));
    element.querySelector('.stage-status').textContent = active && status !== 'loading' ? '진행 중' : statusText(status);
  }
}

// 시계열 지수만 그립니다. 누락된 점을 보간하거나 수익을 만들어 내지 않습니다.
function drawHistory(history) {
  const container = $('#history-chart');
  const fields = [{key:'portfolio',color:'var(--accent)'},{key:'benchmark',color:'#a4acb9'},{key:'rebalance',color:'#b49b7e'}];
  const valid = history.filter(row => fields.some(field => typeof row[field.key] === 'number' && Number.isFinite(row[field.key])));
  if (valid.length < 2) { container.innerHTML = '<p class="empty">실제 검증 데이터가 쌓이면 비교 곡선이 나타납니다.</p>'; return; }
  const values = valid.flatMap(row=>fields.map(f=>row[f.key]).filter(v=>typeof v==='number'&&Number.isFinite(v)));
  const min = Math.min(...values), max = Math.max(...values), range = max-min || 1;
  const width = 600, height = 220, left = 38, right = 10, top = 15, bottom = 30;
  const x = index => left + index / (valid.length-1) * (width-left-right);
  const y = value => top + (max-value)/range*(height-top-bottom);
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="초기값 100으로 비교한 실제 과거 모의운용 성과">`;
  for(let i=0;i<4;i++) { const value = min + range*i/3; const position = y(value); svg += `<line x1="${left}" y1="${position}" x2="${width-right}" y2="${position}" stroke="var(--line)"/><text x="${left-8}" y="${position+3}" text-anchor="end">${safe(number(value,0))}</text>`; }
  for(const field of fields) {
    let path = '', connected = false;
    valid.forEach((row,index)=>{ const value=row[field.key]; if(typeof value !=='number'||!Number.isFinite(value)){connected=false;return;} path += `${connected?'L':'M'}${x(index).toFixed(2)},${y(value).toFixed(2)} `; connected=true; });
    svg += `<path d="${path}" fill="none" stroke="${field.color}" stroke-width="${field.key==='portfolio'?2.8:1.7}" stroke-linejoin="round"/>`;
  }
  svg += `<text x="${left}" y="${height-5}">${safe(valid[0].date)}</text><text x="${width-right}" y="${height-5}" text-anchor="end">${safe(valid[valid.length-1].date)}</text></svg>`;
  container.innerHTML=svg;
}

function calculateScenario() {
  if (!$('#scenario-form').checkValidity()) return;
  const initial = typeof latestState?.paper.current_balance_krw === 'number' ? latestState.paper.current_balance_krw : 1000000;
  if (typeof initial !== 'number' || !Number.isFinite(initial)) { $('#scenario-value').textContent='—'; $('#scenario-note').textContent='현재 평가금액을 확인할 수 없어 계산하지 않았습니다.'; return; }
  const years=Number($('#years').value), monthly=Number($('#contribution').value), rate=Number($('#assumed-return').value)/100;
  const monthlyRate=Math.pow(1+rate,1/12)-1;
  let value=initial;
  for(let month=0;month<years*12;month++) value=value*(1+monthlyRate)+monthly;
  $('#scenario-value').textContent=won(value);
  $('#scenario-note').textContent=`${years}년 · 시작 ${won(initial)} · 매달 ${won(monthly)} 추가 · 가정 연 ${number(rate*100,1)}% · 월말 납입`;
}

async function fetchState() {
  try {
    const response=await fetch('/api/state',{cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(10000)});
    if(!response.ok) throw new Error(`상태 조회 HTTP ${response.status}`);
    render(await response.json());
    $('#connection').textContent='로컬 서버 연결됨 · 일봉 수집과 화면 확인 주기는 서로 다릅니다.';
    $('#connection').classList.remove('error');
  } catch(error) { $('#connection').textContent=`연결 확인 필요 · ${error.message} · 기존 표시값은 마지막 확인값입니다.`; $('#connection').classList.add('error'); }
}

document.querySelectorAll('[data-action]').forEach(button=>button.addEventListener('click',async()=>{
  if(posting)return;
  posting=true;
  document.querySelectorAll('[data-action]').forEach(b=>b.disabled=true);
  try {
    const response=await fetch('/api/jobs',{method:'POST',credentials:'same-origin',headers:{'Content-Type':'application/json','X-Local-Request':'1'},body:JSON.stringify({action:button.dataset.action})});
    const result=await response.json();
    if(!response.ok || result.accepted===false)throw new Error(result.message || result.error || `작업 요청 HTTP ${response.status}`);
    if(result.state)render(result.state);
    $('#connection').textContent=result.message || '작업 요청을 전달했습니다.';
    $('#connection').classList.remove('error');
  }catch(error){$('#connection').textContent=`작업 요청 실패 · ${error.message}`;$('#connection').classList.add('error');}
  finally{posting=false;document.querySelectorAll('[data-action]').forEach(b=>b.disabled=false);}
}));

let theme='light';
try{theme=localStorage.getItem('etf-theme')||'light';}catch{}
function setTheme(value){theme=value==='dark'?'dark':'light';document.documentElement.dataset.theme=theme;$('#theme-toggle').textContent=theme==='dark'?'라이트 모드':'다크 모드';$('#theme-toggle-mobile').textContent=theme==='dark'?'라이트':'다크';$('#theme-toggle-mobile').setAttribute('aria-label',theme==='dark'?'라이트 모드로 전환':'다크 모드로 전환');try{localStorage.setItem('etf-theme',theme);}catch{}}
setTheme(theme);
$('#theme-toggle').addEventListener('click',()=>setTheme(theme==='light'?'dark':'light'));
$('#theme-toggle-mobile').addEventListener('click',()=>setTheme(theme==='light'?'dark':'light'));
function setMotion(){ $('#workflow').classList.toggle('paused',paused); $('#motion-toggle').textContent=paused?'모션 재생':'모션 정지'; $('#motion-toggle').setAttribute('aria-pressed',String(paused)); }
setMotion();
$('#motion-toggle').addEventListener('click',()=>{paused=!paused;setMotion();});
$('#scenario-form').addEventListener('submit',event=>{event.preventDefault();hasScenario=true;calculateScenario();});
document.querySelectorAll('nav a').forEach(link=>link.addEventListener('click',()=>{document.querySelectorAll('nav a').forEach(a=>a.classList.remove('selected'));link.classList.add('selected');}));
// 요청이 겹치지 않게 응답을 받은 다음 1초 뒤에 상태를 다시 확인합니다.
async function poll(){await fetchState();setTimeout(poll,1000);}
poll();
