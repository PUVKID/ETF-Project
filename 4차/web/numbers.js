'use strict';

// 십진수 문자열로 자릿수를 처리합니다. 백분율만 소수 첫째 자리 반올림하며 원값은 바꾸지 않습니다.
(function exposeNumbers(root) {
  function decimalParts(value, shift = 0) {
    const match = String(value).match(/^(-?)(\d+)(?:\.(\d+))?(?:e([+-]?\d+))?$/i);
    if (!match) return null;
    if (Math.abs(Number(match[4] || 0)) > 400) return null;
    const digits = match[2] + (match[3] || '');
    const point = match[2].length + Number(match[4] || 0) + shift;
    const integer = point <= 0 ? '0' : digits.slice(0, point).padEnd(point, '0');
    const fraction = point < 0 ? '0'.repeat(-point) + digits : digits.slice(Math.max(0, point));
    return {negative:match[1] === '-', integer:integer.replace(/^0+(?=\d)/, ''), fraction};
  }

  function truncateDecimal(value, places = 2, shift = 0) {
    if (!Number.isInteger(places) || places < 0 || places > 20 || !Number.isInteger(shift) || Math.abs(shift) > 400) return '—';
    if (typeof value === 'number' && !Number.isFinite(value)) return '—';
    if (typeof value !== 'number' && typeof value !== 'string') return '—';
    const parts = decimalParts(value, shift);
    if (!parts) return '—';
    const fraction = parts.fraction.slice(0, places).padEnd(places, '0');
    const sign = parts.negative && /[1-9]/.test(parts.integer + fraction) ? '-' : '';
    const integer = parts.integer.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return sign + integer + (places ? '.' + fraction : '');
  }

  function number(value, places = 2) {
    return typeof value === 'number' && Number.isFinite(value) ? truncateDecimal(value, places) : '—';
  }

  function percent(value) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
    const parts = decimalParts(value, 2);
    const first = parts.fraction.slice(0, 1).padEnd(1, '0');
    let scaled = BigInt(parts.integer + first);
    if ((parts.fraction[1] || '0') >= '5') scaled += 1n;
    const digits = String(scaled).padStart(2, '0');
    const sign = parts.negative && scaled !== 0n ? '-' : '';
    const integer = digits.slice(0, -1).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    const result = sign + integer + '.' + digits.slice(-1) + '%';
    return result === '100.0%' && value < 1 ? result + ' (반올림)' : result;
  }

  const api = Object.freeze({number, percent, truncateDecimal});
  root.DisplayNumbers = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
