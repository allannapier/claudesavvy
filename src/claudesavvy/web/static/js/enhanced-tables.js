/**
 * Progressive enhancement for data tables: client-side search, column
 * sorting, and pagination.
 *
 * Opt in by adding `data-enhance="table"` to a <table>. Options:
 *   data-page-size="25"   rows per page (default 25)
 *   data-nosort           on a <th> to make that column unsortable
 *
 * Works with HTMX swaps: tables are (re-)enhanced after every settle.
 */
(function () {
    'use strict';

    var SUFFIX_MULTIPLIERS = { k: 1e3, m: 1e6, b: 1e9 };

    function parseCellValue(text) {
        var cleaned = text.trim().replace(/[$,%]/g, '').replace(/,/g, '');
        if (cleaned === '' || cleaned === '-' || cleaned === 'N/A') {
            return { num: null, str: text.trim().toLowerCase() };
        }
        var match = cleaned.match(/^(-?\d+(?:\.\d+)?)\s*([kmb])?$/i);
        if (match) {
            var value = parseFloat(match[1]);
            if (match[2]) value *= SUFFIX_MULTIPLIERS[match[2].toLowerCase()];
            return { num: value, str: cleaned.toLowerCase() };
        }
        return { num: null, str: text.trim().toLowerCase() };
    }

    function getDataRows(tbody) {
        return Array.prototype.filter.call(tbody.rows, function (row) {
            // Exclude empty-state rows that span the whole table
            return !(row.cells.length === 1 && row.cells[0].colSpan > 1);
        });
    }

    function buildToolbar(state) {
        var bar = document.createElement('div');
        bar.className = 'flex flex-wrap items-center justify-between gap-3 px-6 py-3 border-b border-gray-200 bg-gray-50';

        var search = document.createElement('input');
        search.type = 'search';
        search.placeholder = 'Filter rows…';
        search.setAttribute('aria-label', 'Filter table rows');
        search.className = 'w-64 max-w-full px-3 py-1.5 text-sm border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-[#0770E3] focus:border-transparent';
        search.addEventListener('input', function () {
            state.query = search.value.toLowerCase();
            state.page = 0;
            render(state);
        });

        var count = document.createElement('span');
        count.className = 'text-xs text-gray-500';

        bar.appendChild(search);
        bar.appendChild(count);
        state.countEl = count;
        return bar;
    }

    function buildPager(state) {
        var pager = document.createElement('div');
        pager.className = 'flex items-center justify-between gap-3 px-6 py-3 border-t border-gray-200 bg-gray-50';

        var info = document.createElement('span');
        info.className = 'text-xs text-gray-500';

        var nav = document.createElement('div');
        nav.className = 'flex items-center gap-2';

        var btnClass = 'px-3 py-1.5 text-xs font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed';
        var prev = document.createElement('button');
        prev.type = 'button';
        prev.textContent = '← Prev';
        prev.className = btnClass;
        prev.addEventListener('click', function () {
            if (state.page > 0) { state.page--; render(state); }
        });

        var next = document.createElement('button');
        next.type = 'button';
        next.textContent = 'Next →';
        next.className = btnClass;
        next.addEventListener('click', function () {
            state.page++;
            render(state);
        });

        nav.appendChild(prev);
        nav.appendChild(next);
        pager.appendChild(info);
        pager.appendChild(nav);

        state.pagerEl = pager;
        state.pagerInfoEl = info;
        state.prevBtn = prev;
        state.nextBtn = next;
        return pager;
    }

    function applySort(state, rows) {
        if (state.sortCol === null) return rows;
        var col = state.sortCol;
        var dir = state.sortDir;
        return rows.slice().sort(function (a, b) {
            var av = parseCellValue(a.cells[col] ? a.cells[col].textContent : '');
            var bv = parseCellValue(b.cells[col] ? b.cells[col].textContent : '');
            var cmp;
            if (av.num !== null && bv.num !== null) {
                cmp = av.num - bv.num;
            } else if (av.num !== null) {
                cmp = -1;
            } else if (bv.num !== null) {
                cmp = 1;
            } else {
                cmp = av.str < bv.str ? -1 : (av.str > bv.str ? 1 : 0);
            }
            return dir === 'asc' ? cmp : -cmp;
        });
    }

    function render(state) {
        var rows = state.allRows;

        if (state.query) {
            rows = rows.filter(function (row) {
                return row.textContent.toLowerCase().indexOf(state.query) !== -1;
            });
        }

        rows = applySort(state, rows);

        var total = rows.length;
        var pages = Math.max(1, Math.ceil(total / state.pageSize));
        if (state.page >= pages) state.page = pages - 1;
        var start = state.page * state.pageSize;
        var visible = rows.slice(start, start + state.pageSize);

        // Detach all data rows, then re-attach the visible page in order
        state.allRows.forEach(function (row) {
            if (row.parentNode) row.parentNode.removeChild(row);
        });
        visible.forEach(function (row) {
            state.tbody.appendChild(row);
        });

        if (state.emptyRow) {
            state.emptyRow.style.display = visible.length ? 'none' : '';
        }

        state.countEl.textContent = state.query
            ? total + ' of ' + state.allRows.length + ' rows match'
            : state.allRows.length + ' rows';

        if (state.allRows.length <= state.pageSize) {
            state.pagerEl.style.display = 'none';
        } else {
            state.pagerEl.style.display = '';
            state.pagerInfoEl.textContent = total
                ? 'Showing ' + (start + 1) + '–' + Math.min(start + state.pageSize, total) + ' of ' + total
                : 'No matching rows';
            state.prevBtn.disabled = state.page === 0;
            state.nextBtn.disabled = state.page >= pages - 1;
        }

        // Update sort indicators
        state.headers.forEach(function (th, i) {
            var indicator = th.querySelector('.sort-indicator');
            if (!indicator) return;
            indicator.textContent = (i === state.sortCol) ? (state.sortDir === 'asc' ? ' ▲' : ' ▼') : '';
        });
    }

    function enhance(table) {
        if (table.dataset.enhanced === 'true') return;
        var tbody = table.tBodies[0];
        var thead = table.tHead;
        if (!tbody || !thead || !thead.rows.length) return;
        table.dataset.enhanced = 'true';

        var allRows = getDataRows(tbody);
        var emptyRow = Array.prototype.find.call(tbody.rows, function (row) {
            return row.cells.length === 1 && row.cells[0].colSpan > 1;
        }) || null;

        var state = {
            tbody: tbody,
            allRows: allRows,
            emptyRow: emptyRow,
            headers: Array.prototype.slice.call(thead.rows[0].cells),
            query: '',
            sortCol: null,
            sortDir: 'desc',
            page: 0,
            pageSize: parseInt(table.dataset.pageSize, 10) || 25
        };

        // Sortable headers
        state.headers.forEach(function (th, i) {
            if (th.hasAttribute('data-nosort')) return;
            th.classList.add('cursor-pointer', 'select-none', 'hover:text-gray-700');
            th.setAttribute('role', 'button');
            th.setAttribute('tabindex', '0');
            th.setAttribute('aria-label', 'Sort by ' + th.textContent.trim());
            var indicator = document.createElement('span');
            indicator.className = 'sort-indicator';
            th.appendChild(indicator);
            function toggle() {
                if (state.sortCol === i) {
                    state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
                } else {
                    state.sortCol = i;
                    state.sortDir = 'desc';
                }
                render(state);
            }
            th.addEventListener('click', toggle);
            th.addEventListener('keydown', function (e) {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); }
            });
        });

        // Mount toolbar above the table's scroll container and pager below it
        var scrollWrap = table.closest('.overflow-x-auto') || table;
        scrollWrap.parentNode.insertBefore(buildToolbar(state), scrollWrap);
        var pager = buildPager(state);
        if (scrollWrap.nextSibling) {
            scrollWrap.parentNode.insertBefore(pager, scrollWrap.nextSibling);
        } else {
            scrollWrap.parentNode.appendChild(pager);
        }

        render(state);
    }

    function enhanceAll(root) {
        var scope = root && root.querySelectorAll ? root : document;
        var tables = scope.querySelectorAll('table[data-enhance="table"]');
        Array.prototype.forEach.call(tables, enhance);
    }

    document.addEventListener('DOMContentLoaded', function () { enhanceAll(document); });
    document.body.addEventListener('htmx:afterSettle', function (evt) {
        enhanceAll(evt.target);
        // Swapped content may land outside evt.target's subtree
        enhanceAll(document);
    });
})();
