// loader-picker.js â€” Discover and select loading animations without modifying them
// - Scans /loadinganimations/*.html for .loader-card entries
// - Extracts the card title and inner HTML of .loader-wrapper
// - Captures all <style> tags from the source doc and injects them into a shadow root per instance
// - Exposes a ringtone-like picker to choose an animation

(function(){
  function esc(s){
    try{ return String(s||'').replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }catch{ return String(s||''); }
  }
  const SOURCES_DEFAULT = [
    '/loadinganimations/defaultanimations.html',
    '/loadinganimations/coderanimations.html',
    '/loadinganimations/funanimations.html',
    '/loadinganimations/financeanimations.html' // newly supported pack
  ];
  const MANIFEST = '/loadinganimations/manifest.json';
  const LS_KEY_ID = 'loaderAnimationId';
  const LS_KEY_LABEL = 'loaderAnimationLabel';

  let catalog = null; // Array of { id, label, sourceUrl, html, stylesText }
  let stylesCache = new Map(); // sourceUrl -> concatenated styles
  let building = null;

  async function fetchText(url){ const r = await fetch(url, { cache: 'no-store' }); if(!r.ok) throw new Error(`HTTP ${r.status} for ${url}`); return await r.text(); }

  function categoryNameFromSource(sourceUrl){
    try{
      const file = (sourceUrl.split('/').pop()||'').replace(/\.[^.]+$/, '');
      switch(file.toLowerCase()){
        case 'defaultanimations': return 'Default';
        case 'coderanimations': return 'Coder';
        case 'funanimations': return 'Fun';
        case 'financeanimations': return 'Finance';
        default:
          return file.replace(/(^|_|-)([a-z])/g, (m, p1, p2)=> (p1 ? ' ' : '') + p2.toUpperCase()).trim() || 'Other';
      }
    }catch{ return 'Other'; }
  }

  function parseCatalogFromDoc(sourceUrl, htmlText){
    const doc = new DOMParser().parseFromString(htmlText, 'text/html');
    // Gather styles from all <style> tags in source
    const styleEls = Array.from(doc.querySelectorAll('style'));
    const stylesText = styleEls.map(s=> s.textContent||'').join('\n');
    stylesCache.set(sourceUrl, stylesText);
    const category = categoryNameFromSource(sourceUrl);
    const cards = Array.from(doc.querySelectorAll('.loader-card'));
    const entries = cards.map((card, idx)=>{
      const name = (card.querySelector('.loader-title')?.textContent||`Animation ${idx+1}`).trim();
      const wrap = card.querySelector('.loader-wrapper');
      const html = wrap ? wrap.innerHTML : card.innerHTML; // fallback if structure differs
      const id = `${sourceUrl.split('/').pop()||'src'}::${idx}`;
      return { id, name, label: name, sourceUrl, html, stylesText, category };
    });
    return entries;
  }

  async function loadSourcesList(){
    try{
      const t = await fetchText(MANIFEST);
      const list = JSON.parse(t);
      if(Array.isArray(list) && list.length) return list;
    }catch{}
    return SOURCES_DEFAULT;
  }

  async function buildCatalog(){
    if(building) return building;
    building = (async()=>{
      const srcs = await loadSourcesList();
      const entries = [];
      for(const url of srcs){
        try{ const t = await fetchText(url); entries.push(...parseCatalogFromDoc(url, t)); }catch(e){ console.warn('loader-picker: failed to load', url, e); }
      }
      // Keep original encounter order, then relabel numerically 1..N for simple selection
      entries.forEach((e, i)=>{ e.number = i+1; e.label = String(e.number); });
      catalog = entries;
      // Seed default selection if none set: "53 – 23. Quantum Computing"
      try{
        const curId = localStorage.getItem(LS_KEY_ID)||'';
        let curLabel = localStorage.getItem(LS_KEY_LABEL)||'';
        if(!curId && !curLabel){ curLabel = '53 – 23. Quantum Computing'; localStorage.setItem(LS_KEY_LABEL, curLabel); }
        if(!curId && curLabel){
          const m = curLabel.match(/^(\d+)/); let pick = null;
          if(m){ const n = parseInt(m[1],10); pick = catalog.find(x=> x.number===n) || null; }
          if(!pick){ const name = (curLabel.split('–')[1]||'').trim(); if(name) pick = catalog.find(x=> (x.name||'').toLowerCase() === name.toLowerCase()) || null; }
          if(pick) setSelected(pick);
        }
      }catch{}
      return catalog;
    })();
    return building;
  }

  function getSelected(){
    const id = localStorage.getItem(LS_KEY_ID)||'';
    if(!id || !catalog) return null;
    return catalog.find(x=> x.id===id) || null;
  }

  function setSelected(entry){
    try{
      localStorage.setItem(LS_KEY_ID, entry?.id||'');
      const disp = `${entry?.label||''}${entry?.name? ' – ' + entry.name : ''}`.trim();
      localStorage.setItem(LS_KEY_LABEL, disp);
    }catch{}
    // Replace any active loader hosts
    try{ replaceActiveLoaderHosts(); }catch{}
  }

  function getSelectedLabel(){ return localStorage.getItem(LS_KEY_LABEL)||'Default'; }

  function createShadowLoaderFor(entry){
    const host = document.createElement('span');
    host.className = 'ai-loader';
    host.style.display = 'inline-flex';
    host.style.alignItems = 'center';
    host.style.justifyContent = 'center';
    host.style.minHeight = '24px';
    const root = host.attachShadow({ mode:'open' });
    const style = document.createElement('style');
    style.textContent = stylesCache.get(entry.sourceUrl) || entry.stylesText || '';
    root.appendChild(style);
    const wrap = document.createElement('div');
    wrap.innerHTML = entry.html || '';
    root.appendChild(wrap);
    return host;
  }

  // Public: create loader element for current selection, or fallback to existing manager
  function createSelectedLoader(){
    try{
      const sel = getSelected();
      if(sel) return createShadowLoaderFor(sel);
    }catch{}
    // Fallbacks
    if(window.AnimationManager && typeof window.AnimationManager.getLoader==='function'){
      return window.AnimationManager.getLoader();
    }
    const d = document.createElement('div'); d.className='thinking-indicator'; d.innerHTML='<span></span><span></span><span></span>'; return d;
  }

  function replaceActiveLoaderHosts(){
    document.querySelectorAll('.ai-loader-host').forEach(host=>{
      try{
        host.innerHTML = '';
        host.appendChild(createSelectedLoader());
      }catch{}
    });
  }

  // Picker overlay
  async function openPicker(){
    await buildCatalog();
    const ov = document.createElement('div');
    Object.assign(ov.style, { position:'fixed', inset:'0', background:'rgba(0,0,0,0.5)', zIndex: 4000, display:'flex', alignItems:'center', justifyContent:'center' });
    const box = document.createElement('div');
    Object.assign(box.style, { width:'860px', maxWidth:'95vw', maxHeight:'80vh', overflow:'hidden', background:'var(--modal-bg)', border:'1px solid var(--border-color)', borderRadius:'12px', boxShadow:'0 18px 40px rgba(0,0,0,0.35)', display:'flex', flexDirection:'column' });
    box.innerHTML = `<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 14px;border-bottom:1px solid var(--border-color)">
        <div style="font-weight:700">Choose Loading Animation</div>
        <button type="button" id="lpClose" class="btn-pill">Close</button>
      </div>
      <div style="padding:10px; border-bottom:1px solid var(--border-color); display:flex; align-items:center; gap:8px">
        <input id="lpSearch" placeholder="Search by number or category..." style="flex:1; padding:8px 10px; border:1px solid var(--border-color); border-radius:8px; background:transparent;color:var(--text-primary)" />
        <div style="color:var(--text-secondary)">Selected: <span id="lpSel">${getSelectedLabel()}</span></div>
      </div>
      <div id="lpList" style="overflow:auto; padding:10px"></div>`;
    ov.appendChild(box); document.body.appendChild(ov);
    const listEl = box.querySelector('#lpList');
    const searchEl = box.querySelector('#lpSearch');
    const selLabelEl = box.querySelector('#lpSel');
    const close = ()=>{ try{ov.remove();}catch{} };
    box.querySelector('#lpClose').addEventListener('click', close);

    function render(filter=''){
      const q = filter.trim().toLowerCase();
      const isNum = /^\d+$/.test(q);
      const items = catalog.filter(e=>{
        if(!q) return true;
        if(isNum){ return String(e.number||e.label).toLowerCase().startsWith(q); }
        return (e.name||'').toLowerCase().includes(q) || (e.category||'').toLowerCase().includes(q);
      });
      // Group by category
      const groups = new Map();
      for(const e of items){ const k = e.category||'Other'; if(!groups.has(k)) groups.set(k, []); groups.get(k).push(e); }
      const pref = ['Default','Coder','Fun','Finance'];
      const cats = Array.from(groups.keys()).sort((a,b)=>{
        const ia = pref.indexOf(a), ib = pref.indexOf(b);
        if(ia!==-1 || ib!==-1){ return (ia===-1?999:ia) - (ib===-1?999:ib); }
        return a.localeCompare(b);
      });
      listEl.innerHTML = cats.map(cat=>{
        const rows = groups.get(cat).map(e=>{
          const checked = (localStorage.getItem(LS_KEY_ID)===e.id);
          return `<div class="lp-row" data-id="${e.id}" style="display:flex;align-items:center;gap:12px;padding:10px;border-bottom:1px solid var(--border-color);cursor:pointer">
            <div class="ai-loader-host" style="flex:0 0 80px;display:flex;align-items:center;justify-content:center;height:56px;border:1px dashed var(--border-color);border-radius:8px;background:transparent"></div>
            <div style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${e.label}. <span class="lp-name" style="color:var(--text-secondary)">${esc(e.name||'')}</span></div>
            <input type="radio" name="lpPick" ${checked?'checked':''} />
          </div>`;
        }).join('');
        return `<div class="lp-cat" style="position:sticky;top:0;background:var(--modal-bg);z-index:1;padding:6px 8px;font-weight:600;color:var(--text-secondary);border-bottom:1px solid var(--border-color)">${cat}</div>` + rows;
      }).join('');
      // Populate previews via shadow loaders and bind clicks
      listEl.querySelectorAll('.lp-row').forEach(row=>{
        const id = row.getAttribute('data-id');
        const entry = catalog.find(x=> x.id===id);
        const host = row.querySelector('.ai-loader-host');
        try{ host.appendChild(createShadowLoaderFor(entry)); }catch{}
        row.addEventListener('click', ()=>{
          setSelected(entry); selLabelEl.textContent = `${entry.label}${entry.name? ' – ' + entry.name : ''}`; render(filter);
        });
      });
    }

    searchEl.addEventListener('input', ()=> render(searchEl.value||''));
    render('');
  }

  window.LoaderPicker = { buildCatalog, openPicker, createSelectedLoader, replaceActiveLoaderHosts };
})();
