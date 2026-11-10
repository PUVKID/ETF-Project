'use strict';

// 십진수 문자열의 소수점 위치를 옮긴 뒤 자릿수를 자릅니다. 부동소수 곱셈이나 epsilon 보정은 쓰지 않습니다.
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
    return typeof value === 'number' && Number.isFinite(value) ? truncateDecimal(value, 2, 2) + '%' : '—';
  }

  const api = Object.freeze({number, percent, truncateDecimal});
  root.DisplayNumbers = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(globalThis);
