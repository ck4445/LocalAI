(() => {
  const $ = (sel, el=document) => el.querySelector(sel);
  const $$ = (sel, el=document) => Array.from(el.querySelectorAll(sel));
  const fmtGB = v => (v == null ? '—' : `${v} GB`);
  const fmtPct = v => `${Math.round(Math.max(0, Math.min(100, v*100)))}%`;
  const state = {
    catalog: [],          // flat models
    families: [],         // grouped families
    filteredFamilies: [],
    jobs: new Map(),      // jobId -> job
    byModel: new Map(),   // modelId -> {refs}
    byFamily: new Map(),  // familyName -> family object
    activeFilter: 'all',
    search: ''
  };

  // Expose a helper to attach server-returned jobs (e.g., starter packs)
  window.attachStoreJob = function(job){
    try{ attachJob(job, job.name, {}); }catch(e){ /* ignore */ }
  };

  async function loadCatalog() {
    const url = state.search ? `/api/store/models?q=${encodeURIComponent(state.search)}` : '/api/store/models';
    const r = await fetch(url);
    const j = await r.json();
    state.catalog = Array.isArray(j.items) ? j.items : [];
    buildFamilies();
    applyFilter();
  }

  function normFamily(m) {
    if (m.family && m.family.trim()) return m.family.trim();
    // Derive from id
    const id = m.id || '';
    return (id.split(':')[0] || id).trim();
  }

  function pickRecommended(items) {
    // Prefer <= 8B, then <= 12B, else smallest
    let cand = items.filter(x => (x.params_b||0) > 0 && x.params_b <= 8).sort((a,b)=>a.params_b-b.params_b)[0];
    if (!cand) cand = items.filter(x => (x.params_b||0) > 0 && x.params_b <= 12).sort((a,b)=>a.params_b-b.params_b)[0];
    if (!cand) cand = items.filter(x => x.params_b!=null).sort((a,b)=>a.params_b-b.params_b)[0];
    return cand || items[0];
  }

  function buildFamilies() {
    const map = new Map();
    state.catalog.forEach(m => {
      const fam = normFamily(m);
      const key = fam;
      if (!map.has(key)) map.set(key, { name: fam, company: m.company || null, items: [], installedCount: 0, anyMultimodal: false, specialtySet: new Set(), avg: { reasoning: 0, intelligence: 0 } });
      const f = map.get(key);
      f.items.push(m);
      if (m.installed) f.installedCount++;
      if (m.multimodal) f.anyMultimodal = true;
      if (m.specialty) f.specialtySet.add(m.specialty);
    });
    // compute averages
    map.forEach(f => {
      const n = f.items.length;
      const r = f.items.reduce((s, x) => s + (+((x.scores||{}).reasoning||0)), 0);
      const iq = f.items.reduce((s, x) => s + (+((x.scores||{}).intelligence||0)), 0);
      f.avg.reasoning = n ? (r / n).toFixed(1) : '—';
      f.avg.intelligence = n ? (iq / n).toFixed(1) : '—';
      f.specialties = Array.from(f.specialtySet);
      f.recommended = pickRecommended(f.items);
    });
    state.families = Array.from(map.values()).sort((a,b)=>a.name.localeCompare(b.name));
    state.byFamily = map;
  }

  function applyFilter() {
    const f = state.activeFilter;
    const q = (state.search||'').toLowerCase();
    state.filteredFamilies = state.families.filter(F => {
      // Filter by tag
      let tagOk = true;
      if (f === 'multimodal') tagOk = F.anyMultimodal;
      else if (f === 'reasoning') tagOk = F.specialties.some(s=>s.toLowerCase().includes('reason'));
      else if (f === 'embeddings') tagOk = F.specialties.some(s=>s.toLowerCase().includes('embed'));
      else if (f === 'assistant') tagOk = F.specialties.some(s=>s.toLowerCase().includes('assistant'));
      else if (f === 'installed') tagOk = F.installedCount > 0;
      // Search by name/company or any variant id
      const text = `${F.name} ${F.company||''} ${F.items.map(x=>x.id).join(' ')}`.toLowerCase();
      const qOk = !q || text.includes(q);
      return tagOk && qOk;
    });
    renderGrid();
    renderHighlighted();
  }

  function badge(text) {
    const s = document.createElement('span'); s.className = 'badge'; s.textContent = text; return s;
  }

  function renderGrid() {
    const grid = $('#storeGrid');
    grid.innerHTML = '';
    state.filteredFamilies.forEach(F => {
      const card = document.createElement('div'); card.className = 'store-card';
      // Capsule hero
      const cap = document.createElement('div'); cap.className = 'capsule';
      const glow = document.createElement('div'); glow.className = 'glow'; glow.style.background = gradientFor(F.name); cap.appendChild(glow);
      const capTitle = document.createElement('div'); capTitle.style.position='relative'; capTitle.style.zIndex='1'; capTitle.innerHTML = `<div style="font-weight:700;color:#e9ecf7;font-size:18px">${escapeHtml(F.name)}</div><div class="muted" style="font-size:13px">${escapeHtml(F.company||'')}</div>`;
      cap.appendChild(capTitle);
      card.appendChild(cap);
      // Body
      const title = document.createElement('div'); title.className = 'title';
      title.innerHTML = `<span>Recommended: ${escapeHtml(F.recommended?.name || F.recommended?.id || '')}</span>`;
      const subtitle = document.createElement('div'); subtitle.className = 'subtitle';
      subtitle.textContent = `${F.items.length} variants · ${F.installedCount} installed`;
      const badges = document.createElement('div'); badges.className = 'badges'; badges.style.padding='0 14px';
      if (F.anyMultimodal) badges.appendChild(badge('Multimodal'));
      F.specialties.forEach(s => badges.appendChild(badge(s)));
      const metrics = document.createElement('div'); metrics.className = 'metrics';
      const rec = F.recommended || {};
      metrics.innerHTML = `
        <div class="kv">Reasoning: <strong>${F.avg.reasoning}</strong></div>
        <div class="kv">Intelligence: <strong>${F.avg.intelligence}</strong></div>
        <div class="kv">VRAM: <strong>${fmtGB(rec.recommended?.gpu_vram_gb)}</strong></div>
        <div class="kv">Disk: <strong>${fmtGB(rec.recommended?.disk_gb)}</strong></div>
      `;
      const installRow = document.createElement('div'); installRow.className = 'install-row';
      const status = document.createElement('div'); status.className = 'muted'; status.textContent = ' ';
      const prog = document.createElement('div'); prog.className = 'progress'; const bar = document.createElement('div'); bar.className = 'bar'; prog.appendChild(bar); prog.style.display = 'none';
      const spacer = document.createElement('div'); spacer.className = 'spacer';
      const btnInstall = document.createElement('button'); btnInstall.className = 'btn-pill'; btnInstall.type = 'button';
      btnInstall.textContent = (F.recommended?.installed ? 'Installed' : 'Install Recommended');
      btnInstall.disabled = !!F.recommended?.installed;
      btnInstall.addEventListener('click', () => startInstall(F.recommended, { card, bar, btn: btnInstall, status, prog }));
      const btnDetails = document.createElement('button'); btnDetails.className = 'btn-pill'; btnDetails.type = 'button'; btnDetails.textContent = 'View Details';
      btnDetails.addEventListener('click', () => openOverlay(F));
      installRow.appendChild(status); installRow.appendChild(prog); installRow.appendChild(spacer); installRow.appendChild(btnDetails); installRow.appendChild(btnInstall);

      card.appendChild(title);
      card.appendChild(subtitle);
      card.appendChild(badges);
      card.appendChild(metrics);
      card.appendChild(installRow);
      grid.appendChild(card);
      // track recommended variant ui
      if (F.recommended) state.byModel.set(F.recommended.id, { cardEl: card, progressEl: bar, btnEl: btnInstall, statusEl: status, progEl: prog, model: F.recommended });
    });
  }

  function renderHighlighted() {
    const grid = $('#highlightedGrid .store-grid');
    const container = $('#highlightedGrid');
    if (!grid || !container) return;

    const highlightedItems = state.catalog.filter(m => m.highlighted);

    if (highlightedItems.length === 0) {
      container.style.display = 'none';
      return;
    }

    container.style.display = 'block';
    grid.innerHTML = '';

    highlightedItems.forEach(m => {
      const card = document.createElement('div');
      card.className = 'store-card';
      const fam = state.byFamily.get(normFamily(m)) || {};

      card.innerHTML = `
        <div class="capsule">
          <div class="glow" style="background:${gradientFor(m.name)}"></div>
          <div class="cap-text">
            <div class="cap-title">${escapeHtml(m.name)}</div>
            <div class="cap-subtitle">${escapeHtml(m.company || fam.company || '')}</div>
          </div>
        </div>
        <div class="badges" style="padding:0 14px 6px">
          ${m.specialty ? `<span class="badge">${escapeHtml(m.specialty)}</span>` : ''}
          ${m.multimodal ? '<span class="badge">Multimodal</span>' : ''}
          ${m.params_b ? `<span class="badge">${m.params_b}B Params</span>` : ''}
        </div>
        <div class="install-row">
          <div class="spacer"></div>
          <button class="btn-pill" type="button">Install</button>
        </div>
      `;
      const btn = card.querySelector('button');
      btn.addEventListener('click', () => startInstall(m, { btnEl: btn }));
      grid.appendChild(card);
    });
  }

  async function startInstall(m, refs) {
    const btn = refs.btn || refs.btnEl;
    btn.disabled = true; btn.textContent = 'Queued…';
    refs.statusEl.textContent = 'Queued';
    refs.progEl.style.display = '';
    try {
      const r = await fetch('/api/store/pull', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: m.id }) });
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();
      const job = j; // contains id, name, status, etc.
      attachJob(job, m.id, refs);
    } catch (e) {
      refs.statusEl.textContent = 'Failed to queue';
      btn.disabled = false; btn.textContent = 'Install';
    }
  }

  function attachJob(job, modelId, refs) {
    state.jobs.set(job.id, { ...job, modelId, refs });
    addToDock(job);
    streamJob(job.id, ev => onJobEvent(job.id, ev));
  }

  function addToDock(job) {
    const dock = $('#downloadsDock');
    const list = $('#downloadsList');
    dock.classList.add('visible');
    const item = document.createElement('div'); item.className = 'dl-item'; item.dataset.jobId = job.id;
    item.innerHTML = `
      <div class="row"><div class="name">${escapeHtml(job.name)}</div><div style="margin-left:auto" class="muted" data-role="msg">${escapeHtml(job.message||'Queued')}</div></div>
      <div class="progress" style="margin-top:6px"><div class="bar" style="width:${fmtPct(job.progress||0)}"></div></div>
      <div class="row" style="margin-top:6px"><div class="muted" data-role="detail"></div><div style="margin-left:auto"><button class="btn-pill" data-action="cancel" type="button">Cancel</button></div></div>
    `;
    list.prepend(item);
    item.querySelector('[data-action="cancel"]').addEventListener('click', async () => {
      try { await fetch(`/api/store/jobs/${job.id}/cancel`, { method: 'POST' }); } catch {}
      // Optimistically remove the dock item and hide the dock if empty
      removeDockItem(job.id);
    });
  }

  function onJobEvent(jobId, ev) {
    const job = state.jobs.get(jobId) || {}; Object.assign(job, ev.data || {}); state.jobs.set(jobId, job);
    // Update dock
    const item = $(`.dl-item[data-job-id="${jobId}"]`);
    if (item) {
      const bar = $('.bar', item.parentElement ? item : item);
      if (bar) bar.style.width = fmtPct(job.progress || 0);
      const msg = item.querySelector('[data-role="msg"]'); if (msg) msg.textContent = job.message || job.status || ev.event;
      const detail = item.querySelector('[data-role="detail"]');
      if (detail) {
        const bc = job.bytes_completed || 0; const bt = job.bytes_total || 0;
        detail.textContent = bt ? `${fmtBytes(bc)} / ${fmtBytes(bt)} (${fmtPct((job.progress||0))})` : '';
      }
    }
    // Remove dock item on terminal events and close overlay for canceled
    if (ev.event === 'canceled' || ev.event === 'completed' || ev.event === 'error') {
      removeDockItem(jobId);
      if (ev.event === 'canceled') {
        const ov = document.getElementById('storeOverlay');
        if (ov) ov.classList.remove('visible');
      }
    }
    // Update card
    const ui = (state.byModel.get(job.modelId || job.name) || state.byModel.get(job.name) || {});
    const r = job.refs || {};
    const barCard = ui.progressEl || r.progressEl;
    const statusEl = ui.statusEl || r.statusEl;
    const btnEl = ui.btnEl || r.btnEl;
    const progEl = ui.progEl || r.progEl;
    if (barCard) barCard.style.width = fmtPct(job.progress || 0);
    if (statusEl) statusEl.textContent = job.message || job.status || '';
    if (progEl) progEl.style.display = '';
    if (job.status === 'completed') {
      if (btnEl) { btnEl.textContent = 'Installed'; btnEl.disabled = true; }
      // mark installed in local state
      const entry = state.byModel.get(job.modelId || job.name) || state.byModel.get(job.name);
      if (entry) entry.model.installed = true;
    }
  }

  async function streamJob(jobId, onEvent) {
    try {
      const res = await fetch(`/api/store/jobs/${jobId}/stream`);
      if (!res.ok || !res.body) return;
      const reader = res.body.getReader();
      const td = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += td.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, idx).trim(); buf = buf.slice(idx+1);
          if (!line) continue;
          try { const obj = JSON.parse(line); onEvent(obj); } catch {}
        }
      }
        } catch (e) {
      console.error(`[store] Stream for job ${jobId} failed:`, e);
      onEvent({ event: 'error', data: { id: jobId, error: 'Stream disconnected', status: 'error', message: 'Stream disconnected' } });
    }
  }

  function attachUI() {
    $('#searchInput').addEventListener('input', e => { state.search = e.target.value || ''; debounceReload(); });
    $$('.store-filters .chip').forEach(chip => chip.addEventListener('click', () => {
      $$('.store-filters .chip').forEach(c => c.setAttribute('aria-pressed', 'false'));
      chip.setAttribute('aria-pressed', 'true');
      state.activeFilter = chip.dataset.filter || 'all';
      applyFilter();
    }));
    $('#ovClose').addEventListener('click', closeOverlay);
    // Polling for job status has been removed in favor of the real-time EventSource stream.
    // The streamJob function now handles all job state updates.
  }

  let reloadTimer;
  function debounceReload() { clearTimeout(reloadTimer); reloadTimer = setTimeout(loadCatalog, 200); }

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c])); }
  function fmtBytes(n) {
    if (!n || n < 0) return '0 B';
    const u = ['B','KB','MB','GB','TB'];
    let i = 0; let v = n;
    while (v >= 1024 && i < u.length-1) { v /= 1024; i++; }
    return `${v.toFixed(v<10&&i>0?1:0)} ${u[i]}`;
  }

  // init
  attachUI();
  loadCatalog();

  // Overlay and family details
  function openOverlay(F) {
    const ov = $('#storeOverlay');
    $('#ovTitle').textContent = F.name;
    $('#ovSubtitle').textContent = `${F.company || '—'} · ${F.items.length} variants`;
    const glow = $('#ovGlow'); glow.style.background = gradientFor(F.name);
    const badges = $('#ovBadges'); badges.innerHTML='';
    if (F.anyMultimodal) badges.appendChild(badge('Multimodal'));
    F.specialties.forEach(s => badges.appendChild(badge(s)));
    badges.appendChild(badge(`Avg Reasoning ${F.avg.reasoning}`));
    badges.appendChild(badge(`Avg Intelligence ${F.avg.intelligence}`));
    // Variants table
    const box = $('#ovVariants'); box.innerHTML='';
    const header = document.createElement('div'); header.className = 'row header';
    header.innerHTML = '<div>Variant</div><div>Reasoning</div><div>Intelligence</div><div>Recommended Specs</div><div></div>';
    box.appendChild(header);
    F.items.slice().sort((a,b)=> (a.params_b||0)-(b.params_b||0)).forEach(m => {
      const row = document.createElement('div'); row.className = 'row';
      row.style.cursor = 'pointer';
      const specs = `${fmtGB(m.recommended?.gpu_vram_gb)} VRAM · ${fmtGB(m.recommended?.disk_gb)} Disk`;
      row.innerHTML = `
        <div><strong>${escapeHtml(m.name||m.id)}</strong><div class="muted" style="font-size:12px">${escapeHtml(m.id)}</div></div>
        <div>${m.scores?.reasoning ?? '—'}</div>
        <div>${m.scores?.intelligence ?? '—'}</div>
        <div>${specs}</div>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="progress" style="width:120px;display:none"><div class="bar"></div></div>
          <button class="btn-pill" type="button">${m.installed ? 'Installed' : 'Install'}</button>
        </div>
      `;
      const prog = row.querySelector('.progress');
      const bar = row.querySelector('.bar');
      const btn = row.querySelector('button');
      const statusEl = document.createElement('div'); statusEl.className='muted'; statusEl.style.marginLeft='10px'; statusEl.style.fontSize='12px';
      row.lastElementChild.appendChild(statusEl);
      if (m.installed) btn.disabled = true;
      btn.addEventListener('click', () => startInstall(m, { progressEl: bar, progEl: prog, btn, statusEl }));
      // Detail expander row
      const det = document.createElement('div'); det.className = 'row'; det.style.display='none'; det.style.gridTemplateColumns='1fr'; det.style.background='#0b1219';
      det.innerHTML = `
        <div class="muted" style="font-size:13px; line-height:1.6">
          <div><strong>Model ID:</strong> <code>${escapeHtml(m.id)}</code></div>
          <div><strong>Family:</strong> ${escapeHtml(m.family || F.name)} · <strong>Company:</strong> ${escapeHtml(m.company || F.company || '—')}</div>
          <div><strong>Multimodal:</strong> ${m.multimodal ? 'Yes' : 'No'} · <strong>Specialty:</strong> ${escapeHtml(m.specialty || '—')}</div>
          <div><strong>Recommended:</strong> ${fmtGB(m.recommended?.gpu_vram_gb)} VRAM, ${fmtGB(m.recommended?.system_ram_gb)} RAM, ${fmtGB(m.recommended?.disk_gb)} Disk</div>
          <div><button class="btn-pill" type="button" data-copy="${escapeHtml(m.id)}">Copy ID</button></div>
        </div>`;
      // toggle details when clicking anywhere except the install button
      row.addEventListener('click', (ev) => {
        if (ev.target === btn || btn.contains(ev.target)) return;
        det.style.display = det.style.display === 'none' ? 'grid' : 'none';
      });
      det.addEventListener('click', (ev) => {
        const b = ev.target.closest('[data-copy]'); if (!b) return;
        const txt = b.getAttribute('data-copy');
        navigator.clipboard?.writeText(txt).then(()=>{ b.textContent = 'Copied'; setTimeout(()=>b.textContent='Copy ID', 1000); });
      });
      box.appendChild(row);
      box.appendChild(det);
      state.byModel.set(m.id, { progressEl: bar, progEl: prog, btnEl: btn, statusEl, model: m });
    });
    ov.classList.add('visible');
  }
  function closeOverlay() { $('#storeOverlay').classList.remove('visible'); }

  function gradientFor(name) {
    const h = hash(name) % 360; const h2 = (h+60)%360;
    const a = `linear-gradient(120deg, hsl(${h} 70% 22%), hsl(${h2} 70% 12%))`;
    return a;
  }
  function hash(s) { let h=0; for (let i=0;i<s.length;i++) { h=((h<<5)-h)+s.charCodeAt(i); h|=0; } return Math.abs(h); }
  function removeDockItem(jobId) {
    const el = document.querySelector(`.dl-item[data-job-id="${jobId}"]`);
    if (el && el.parentElement) el.parentElement.removeChild(el);
    const list = document.getElementById('downloadsList');
    const dock = document.getElementById('downloadsDock');
    if (list && list.children.length === 0 && dock) dock.classList.remove('visible');
  }
})();
