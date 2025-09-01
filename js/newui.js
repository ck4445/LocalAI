// newui.js â€” Full wiring for newui.html without changing its look
// Uses backend endpoints in app.py. Keeps newui styles/layout intact.

(function(){
  // --- State ---
  let currentSettings = null;
  let models = [];
  let modelMeta = {};
  let currentModel = null;
  let useFriendlyNames = (localStorage.getItem('useFriendlyModelNames') !== '0');
  let currentChatId = null;
  let isStreaming = false;
  let streamCtrl = null;
  let currentSID = null;
  let currentDraftId = null;
  let currentAttachments = [];
  let currentAttachmentsTokens = 0;
  let searchData = [];
  let isShutdown = false;
  let tts = { utter: null, speaking: false, paused: false };

  // --- DOM ---
  const qs = (s, root=document)=>root.querySelector(s);
  const qsa = (s, root=document)=>Array.from(root.querySelectorAll(s));
  const byId = (id)=>document.getElementById(id);

  const els = {
    // Header / model
    modelBtn: byId('model-selector-btn'),
    modelPop: byId('model-popup'),
    modelName: byId('model-name'),

    // Sidebar
    newChatBtn: byId('new-chat-btn'),
    userProfileBtn: byId('user-profile-btn'),
    sidebarMiddle: qs('.sidebar-middle'),

    // Chat
    chatView: byId('chat-view'),
    chatContainer: byId('chat-container'),
    chatContent: byId('chat-content'),
    welcomeForm: byId('welcome-chat-form'),
    welcomeInput: byId('welcome-chat-input'),

    // Settings modal
    settingsOverlay: byId('settings-modal-overlay'),
    settingsCloseBtn: byId('settings-close-btn'),
    settingsSidebar: null, // set when rendered
    settingsBody: null,    // set when rendered

    // Search
    searchBtn: byId('search-chats-btn'),
    searchOverlay: byId('search-overlay'),
    searchInput: byId('search-input'),
    searchResults: byId('search-results'),

    // File input
    fileInput: byId('global-file-input'),
    // Composer areas (may or may not exist at boot)
    mainFooter: byId('main-footer'),
    mainForm: byId('main-chat-form'),
    mainInput: byId('main-chat-input'),
  };

  // --- Helpers ---
  const escapeHTML = (s)=> (window.escapeHTML ? window.escapeHTML(s) : String(s||'')
    .replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c])));
  const markedParse = (md)=> window.marked ? marked.parse(md) : escapeHTML(md);
  const renderMarkdownSafe = (container)=> {
    try{ if(window.renderMarkdown) renderMarkdown(container); }catch{}
  };

  // --- Actions hook (delegated to index.html) ---
  window.attachActions = function(wrapper){
    try{
      if(!wrapper) return;
      // Mark finalized for UI scripts and dispatch an event so the
      // icon-only action bar in index.html can attach controls.
      wrapper.dataset.final = '1';
      window.dispatchEvent(new CustomEvent('attach-actions', { detail: { node: wrapper } }));
    }catch{}
  }
  // --- Experimental: confetti + tool commands ---
  let __confettiCSS = false;
  function ensureConfettiStyles(){
    if(__confettiCSS) return; __confettiCSS = true;
    const style = document.createElement('style');
    style.textContent = `@keyframes nu-confetti-fall {0%{transform:translateY(-10vh) rotate(0)}100%{transform:translateY(100vh) rotate(720deg)}}`;
    document.head.appendChild(style);
  }
  function triggerConfetti(durationMs=2000, count=80){
    ensureConfettiStyles();
    const wrap = document.createElement('div');
    Object.assign(wrap.style, {position:'fixed',left:0,top:0,width:'100%',height:'0',overflow:'visible',pointerEvents:'none',zIndex:9999});
    document.body.appendChild(wrap);
    const colors=['#ff6b6b','#ffd93d','#6bcBef','#51cf66','#845ef7','#f783ac','#ffa94d'];
    for(let i=0;i<count;i++){
      const p=document.createElement('div');
      const size=6+Math.random()*6;
      const left=Math.random()*100;
      const delay=Math.random()*0.4;
      const dur=1.6+Math.random()*0.9;
      Object.assign(p.style,{position:'absolute',left:left+'%',top:'-10px',width:size+'px',height:size+'px',background:colors[i%colors.length],opacity:0.9,transform:'translateY(-10vh)',animation:`nu-confetti-fall ${dur}s linear ${delay}s forwards`});
      wrap.appendChild(p);
    }
    setTimeout(()=>{ try{wrap.remove();}catch{} }, durationMs);
  }
  function setShutdownOverlay(on){
    isShutdown = !!on;
    let ov = byId('nu-shutdown-overlay');
    if(on){
      if(!ov){
        ov = document.createElement('div');
        ov.id = 'nu-shutdown-overlay';
        Object.assign(ov.style,{position:'fixed',inset:0,background:'rgba(0,0,0,0.6)',backdropFilter:'blur(2px)',display:'flex',alignItems:'center',justifyContent:'center',color:'#fff',zIndex:10000});
        ov.innerHTML = `<div style="text-align:center"><div style="font-size:22px;font-weight:800;margin-bottom:8px">Assistant is sleeping</div><div style="opacity:0.9">Click anywhere to wake</div></div>`;
        ov.addEventListener('click', ()=> setShutdownOverlay(false));
        document.body.appendChild(ov);
      }
    } else {
      try{ ov?.remove(); }catch{}
    }
  }
  function extractToolCommands(text){
    const lines = (text||'').split(/\r?\n/);
    const tools=[]; let cutoff = lines.length;
    for(let i=lines.length-1;i>=0;i--){
      const m = lines[i].match(/^!([a-zA-Z][\w-]*)(?:\s+(.+))?$/);
      if(m){ tools.unshift({cmd:m[1].toLowerCase(), arg:(m[2]||'').trim()}); cutoff = i; }
      else if(lines[i].trim()==='') { cutoff = Math.min(cutoff, i); }
      else break;
    }
    return { clean: lines.slice(0, cutoff).join('\n'), tools };
  }
  async function runToolCommands(tools){
    if(!tools || !tools.length) return;
    for(const t of tools){
      switch(t.cmd){
        case 'celebrate': triggerConfetti(); break;
        case 'shutdown': setShutdownOverlay(true); break;
        case 'ignore': toast('Tool requested: ignore'); break;
        case 'note': if(t.arg) toast(t.arg); break;
        case 'theme': if(t.arg && /^(dark|light)$/i.test(t.arg)){ document.documentElement.dataset.theme = t.arg.toLowerCase(); } break;
        // removed
        default: break;
      }
    }
  }
  const estimateTokens = (t)=> (window.estimateTokens ? window.estimateTokens(t) : Math.max(1, ((t||'').length+3)>>2));
  const ALLOWED_EXTS = new Set(['.txt','.md','.markdown','.py','.js','.ts','.json','.html','.htm','.css','.c','.cc','.cpp','.h','.hpp','.java','.cs','.rs','.go','.rb','.php','.sh','.bash','.zsh','.yaml','.yml','.toml','.ini','.cfg','.conf','.env','.sql','.xml','.tex','.r','.kt','.swift','.pl','.lua','.hs','.m','.mm','.ps1','.clj','.scala','.tsx','.jsx']);
  const PER_FILE_TOKEN_LIMIT = 20000, TOTAL_TOKEN_LIMIT = 25000;

  const setChatEmpty = (isEmpty)=>{
    els.chatView.classList.toggle('is-empty', !!isEmpty);
  };
  const scrollToBottom = ()=>{ try{ els.chatContainer.scrollTop = els.chatContainer.scrollHeight; }catch{} };

  // --- API ---
  async function loadUser(){
    try{
      const r = await fetch('/api/user');
      const data = await r.json();
      const name = (data.name||'User').trim();
      // Update sidebar user display
      const avatar = qs('.user-avatar'); if(avatar) avatar.textContent = (name||'U').slice(0,1).toUpperCase();
      const uname = qs('.user-name'); if(uname) uname.textContent = name;
    }catch(e){ console.error('loadUser failed', e); }
  }

  async function loadSettings(){
    try{
      const r = await fetch('/api/settings');
      const data = await r.json();
      currentSettings = data.settings || {};
      // Personality library removed
    }catch(e){ console.error('loadSettings failed', e); }
  }

  async function saveSettings(next){
    try{
      currentSettings = Object.assign({}, currentSettings||{}, next||{});
      await fetch('/api/settings',{ method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ settings: currentSettings }) });
    }catch(e){ console.error('saveSettings failed', e); }
  }

  async function loadModels(){
    try{
      const r = await fetch('/api/models');
      const data = await r.json();
      models = data.models || [];
      modelMeta = data.meta || {};
      if(!currentModel && models.length){
        const pref = currentSettings?.default_model || null;
        currentModel = (pref && models.includes(pref)) ? pref : models[0];
      }
      updateModelHeader();
      renderModelsPopup();
    }catch(e){ console.error('loadModels failed', e); }
  }

  async function refreshChats(){
    try{
      const r = await fetch('/api/chats');
      const data = await r.json();
      const list = data.chats || [];
      renderChatsList(list);
    }catch(e){ console.error('refreshChats failed', e); }
  }

  async function newChat(){
    try{
      const r = await fetch('/api/chats', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ title: 'New chat' })});
      const data = await r.json();
      currentChatId = data.id;
      els.chatContent.innerHTML='';
      setChatEmpty(true);
      if(els.welcomeInput){ els.welcomeInput.value=''; resizeTextarea(els.welcomeInput); }
      await refreshChats();
    }catch(e){ console.error('newChat failed', e); }
  }

  async function loadChat(id){
    try{
      const r = await fetch(`/api/chats/${id}`);
      if(!r.ok) throw new Error(`HTTP ${r.status}`);
      const chat = await r.json();
      currentChatId = chat.id;
      els.chatContent.innerHTML='';
      for(const m of (chat.messages||[])){
        if(m.role === 'user') addUserMessage(m.content);
        else addAssistantMessage(m.content, true);
      }
      const empty = !((chat.messages||[]).length);
      setChatEmpty(empty);
      updateComposerVisibility(empty);
      await fetchAttachments();
      scrollToBottom();
    }catch(e){ console.error('loadChat failed', e); toast('Failed to load chat'); }
  }

  // --- Chat rendering ---
  function addUserMessage(text){
    const html = `<div class="chat-message user-message"><div class="message-bubble">${escapeHTML(text).replace(/\n/g,'<br>')}</div></div>`;
    els.chatContent.insertAdjacentHTML('beforeend', html);
    scrollToBottom();
  }
  function addAssistantMessage(markdownText, finalize=false){
    const wrapper = document.createElement('div');
    wrapper.className = 'chat-message ai-message';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    wrapper.appendChild(bubble);
    if(markdownText){
      bubble.innerHTML = markedParse(markdownText);
      renderMarkdownSafe(bubble);
    }else{
      // thinking indicator using selected loader (fallback to dots)
      bubble.innerHTML = '';
      const host = document.createElement('div');
      host.className = 'ai-loader-host';
      try{
        const node = (window.LoaderPicker && typeof window.LoaderPicker.createSelectedLoader==='function')
          ? window.LoaderPicker.createSelectedLoader()
          : null;
        if(node) host.appendChild(node);
        else host.innerHTML = `<div class="thinking-indicator"><span></span><span></span><span></span></div>`;
      }catch{ host.innerHTML = `<div class=\"thinking-indicator\"><span></span><span></span><span></span></div>`; }
      bubble.appendChild(host);
    }
    els.chatContent.appendChild(wrapper);
    scrollToBottom();
    if (finalize) { try { if (window.attachActions) window.attachActions(wrapper); } catch {} }
    return bubble;
  }

  // --- Composer visibility ---
  function updateComposerVisibility(isEmpty){
    const footer = byId('main-footer');
    if(!footer) return;
    footer.style.display = isEmpty ? 'none' : '';
    // Re-render attachment strip so it moves to the visible composer
    try{ renderAttachmentChips(); }catch{}
  }

  // --- Attachment UI ---
  function fileTypeFromName(name){
    try{ const ext = (name.split('.').pop()||'').toLowerCase(); return ext || 'file'; }catch{ return 'file'; }
  }
  function fileTypeLabel(name){
    const ext = fileTypeFromName(name);
    if(ext === 'pdf') return 'PDF';
    return ext ? ext.toUpperCase() : 'FILE';
  }
  function iconColorForExt(ext){
    switch(ext){
      case 'pdf': return '#ff6b9e';
      case 'doc': case 'docx': return '#4f8cff';
      case 'ppt': case 'pptx': return '#ff7a59';
      case 'xls': case 'xlsx': return '#3bb273';
      case 'html': case 'htm': return '#4ea1ff';
      case 'md': case 'txt': return '#9da3af';
      default: return '#4ea1ff';
    }
  }
  function ensureAttachmentHost(){
    // Create a strip above the currently visible input area (welcome or main)
    const isEmpty = !!els.chatView?.classList?.contains('is-empty');
    const desiredParent = (
      isEmpty
        ? document.querySelector('.welcome-screen .input-area-container')
        : document.querySelector('#main-footer .input-area-container')
    ) || els.mainForm || document.getElementById('main-chat-form') || document.getElementById('chat-view') || document.body;

    let host = document.getElementById('attachment-strip');
    if(!host){
      host = document.createElement('div');
      host.id = 'attachment-strip';
      Object.assign(host.style, { display: 'none', padding: '8px 8px 12px 8px' });
      // Delegate remove handler (once)
      host.addEventListener('click', async (e)=>{
        const btn = e.target.closest('[data-remove]');
        if(!btn) return;
        e.preventDefault(); e.stopPropagation();
        const id = btn.getAttribute('data-remove');
        if(!(id && currentChatId && currentDraftId)) return;
        try{
          const res = await fetch(`/api/chats/${currentChatId}/drafts/${currentDraftId}/attachments/${id}`, { method:'DELETE' });
          if(res.ok) await fetchAttachments();
        }catch(err){ console.error('remove attachment failed', err); }
      });
    }
    // Reparent if needed so itâ€™s always above the visible composer
    if(host.parentElement !== desiredParent){
      if(desiredParent && desiredParent.firstChild){ desiredParent.insertBefore(host, desiredParent.firstChild); }
      else if(desiredParent){ desiredParent.appendChild(host); }
    }
    return host;
  }
  function renderAttachmentChips(){
    const host = ensureAttachmentHost(); if(!host) return;
    if(!currentAttachments || currentAttachments.length===0){
      host.style.display = 'none'; host.innerHTML = ''; return;
    }
    host.style.display = '';
    const isLight = (document.documentElement.dataset.theme||'dark')==='light';
    const cardBg = isLight ? '#f6f7fb' : 'transparent';
    const border = isLight ? 'rgba(0,0,0,0.12)' : 'var(--border-color)';
    const chips = currentAttachments.map(a=>{
      const ext = fileTypeFromName(a.name);
      const color = iconColorForExt(ext);
      const label = fileTypeLabel(a.name);
      return `<div class="att-chip" data-att="${a.id}" style="position:relative; display:flex; align-items:center; gap:12px; padding:10px 12px; border:1px solid ${border}; border-radius:12px; background:${cardBg}; min-width:260px; max-width:360px">
        <div style="width:28px;height:28px;border-radius:8px;background:${color};display:flex;align-items:center;justify-content:center;color:white;font-weight:700">${label.slice(0,1)}</div>
        <div style="flex:1;min-width:0;overflow:hidden">
          <div style="font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis">${escapeHTML(a.name)}</div>
          <div style="font-size:12px;color:var(--text-secondary)">${escapeHTML(label)}</div>
        </div>
        <button type="button" title="Remove" aria-label="Remove" data-remove="${a.id}" style="position:absolute; right:6px; top:6px; width:20px; height:20px; border-radius:50%; border:1px solid ${border}; background:${isLight?'#ffffff':'#2B2B2B'}; color:var(--text-primary); display:flex; align-items:center; justify-content:center; cursor:pointer">Ã—</button>
      </div>`;
    }).join('');
    host.innerHTML = `<div style="display:flex; flex-wrap:wrap; gap:12px; align-items:center">${chips}</div>`;
  }
  async function fetchAttachments(){
    if(!currentChatId){ renderAttachmentChips(); return; }
    try{
      if(!currentDraftId){
        // Probe latest draft created by any UI
        const r0 = await fetch(`/api/chats/${currentChatId}/drafts/latest`);
        const d0 = await r0.json();
        if(d0?.draft_id){ currentDraftId = d0.draft_id; currentAttachments = d0.items||[]; currentAttachmentsTokens = d0.total_tokens||0; }
      }
      if(currentDraftId){
        const r = await fetch(`/api/chats/${currentChatId}/drafts/${currentDraftId}/attachments`);
        const data = await r.json();
        currentAttachments = data.items || currentAttachments || [];
        currentAttachmentsTokens = data.total_tokens || currentAttachmentsTokens || 0;
      }
    }catch(e){ console.error('fetchAttachments failed', e); currentAttachments = []; }
    renderAttachmentChips();
  }

  // --- Drag & Drop onto composer ---
  function bindDragAndDrop(){
    const targets = [
      document.querySelector('.main-footer .input-area-container .input-wrapper'),
      document.querySelector('.input-area-container .input-wrapper'),
      document.querySelector('.input-wrapper'),
      els.mainInput,
    ].filter(Boolean);
    // Prevent default browser open-on-drop
    ['dragover','drop'].forEach(ev=> document.addEventListener(ev, (e)=>{ e.preventDefault(); }, false));
    targets.forEach(el=>{
      ['dragenter','dragover'].forEach(ev=> el.addEventListener(ev, (e)=>{ e.preventDefault(); e.dataTransfer.dropEffect='copy'; el.style.outline='2px dashed #4ea1ff'; el.style.outlineOffset='4px'; }, false));
      ['dragleave','dragend','drop'].forEach(ev=> el.addEventListener(ev, ()=>{ el.style.outline=''; el.style.outlineOffset=''; }, false));
      el.addEventListener('drop', async (e)=>{
        e.preventDefault();
        const files = Array.from(e.dataTransfer?.files||[]);
        if(files.length){ await handleFiles(files); }
      });
    });
  }

  // --- Streaming ---
  function genSID(){
    if(crypto && crypto.randomUUID) return crypto.randomUUID();
    return 'sid_'+Math.random().toString(36).slice(2);
  }
  async function streamChatSend(text){
    if(isShutdown){ toast('Assistant is sleeping. Click to wake.'); return; }
    if(!currentModel){ await loadModels(); if(!currentModel){ toast('No models available'); return; } }
    // ensure chat exists
    if(!currentChatId){
      try{
        const title = (text||'').trim().split(/\s+/).slice(0,3).join(' ')||'New chat';
        const r = await fetch('/api/chats', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ title })});
        const created = await r.json();
        currentChatId = created.id; await refreshChats();
      }catch(e){ console.error('create chat failed', e); toast('Could not create chat'); return; }
    }
    // hide welcome composer once we have a message
    setChatEmpty(false);
    updateComposerVisibility(false);
    addUserMessage(text);
    const bubble = addAssistantMessage('', false);
    isStreaming = true; currentSID = genSID();
    try{
      streamCtrl = new AbortController();
      const res = await fetch('/api/chat/stream', {
        method:'POST', headers:{'Content-Type':'application/json'}, signal: streamCtrl.signal,
        body: JSON.stringify({ chat_id: currentChatId, model: currentModel, user_message: text, sid: currentSID, draft_id: currentDraftId||null })
      });
      if(!res.ok){
        const detail = await res.text().catch(()=> '');
        bubble.innerHTML = `<div style="color:#ffb4b4">Error (${res.status})<br/><small>${escapeHTML(detail.slice(0,500))}</small></div>`;
        return;
      }
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      if(!reader){ bubble.textContent = 'Stream unavailable.'; return; }
      let ndjsonBuffer = '';
      let assistantBuffer = '';
      while(true){
        const { value, done } = await reader.read();
        if(done) break;
        ndjsonBuffer += decoder.decode(value, { stream: true });
        const lines = ndjsonBuffer.split('\n');
        ndjsonBuffer = lines.pop() || '';
        for(const line of lines){
          if(!line.trim()) continue;
          let payload; try{ payload = JSON.parse(line); }catch{ continue; }
          const msg = payload.message || {};
          if(typeof msg.content === 'string' && msg.content){
            assistantBuffer += msg.content;
            bubble.innerHTML = markedParse(assistantBuffer.trim());
            renderMarkdownSafe(bubble);
            scrollToBottom();
          }
          if(payload.done){
            if(payload.error){
              assistantBuffer += `\n\n*Error: ${payload.error}*`;
              bubble.innerHTML = markedParse(assistantBuffer.trim());
              renderMarkdownSafe(bubble);
            }
          }
        }
      }
      // Experimental tool commands removed
      try {
        const wrap = bubble.parentElement;
        if(wrap){
          wrap.dataset.final = '1';
          wrap.dataset.regenerating = '';
          if(window.attachActions) window.attachActions(wrap);
        }
      } catch {}
    }catch(e){
      bubble.innerHTML = `<div style="color:#ffb4b4">${escapeHTML(e?.message||'Stream error')}</div>`;
    }finally{
      isStreaming = false; currentSID = null; streamCtrl = null;
    }
  }

  // Stream directly into an existing assistant bubble (for regenerate)
  async function streamIntoBubble(text, bubble){
    if(!bubble) return;
    if(isShutdown){ toast('Assistant is sleeping. Click to wake.'); return; }
    if(!currentModel){ await loadModels(); if(!currentModel){ toast('No models available'); return; } }
    if(!currentChatId){
      try{
        const title = (text||'').trim().split(/\s+/).slice(0,3).join(' ')||'New chat';
        const r = await fetch('/api/chats', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ title })});
        const created = await r.json();
        currentChatId = created.id; await refreshChats();
      }catch(e){ console.error('create chat failed', e); toast('Could not create chat'); return; }
    }
    isStreaming = true; currentSID = genSID();
    try{
      streamCtrl = new AbortController();
      const res = await fetch('/api/chat/stream', {
        method:'POST', headers:{'Content-Type':'application/json'}, signal: streamCtrl.signal,
        body: JSON.stringify({ chat_id: currentChatId, model: currentModel, user_message: text, sid: currentSID, draft_id: currentDraftId||null })
      });
      if(!res.ok){ const detail = await res.text().catch(()=> ''); bubble.innerHTML = `<div style="color:#ffb4b4">Error (${res.status})<br/><small>${escapeHTML(detail.slice(0,500))}</small></div>`; return; }
      const reader = res.body?.getReader(); const decoder = new TextDecoder(); if(!reader){ bubble.textContent = 'Stream unavailable.'; return; }
      let ndjsonBuffer = ''; let assistantBuffer = '';
      while(true){
        const { value, done } = await reader.read();
        if(done) break;
        ndjsonBuffer += decoder.decode(value, { stream: true });
        const lines = ndjsonBuffer.split('\n'); ndjsonBuffer = lines.pop() || '';
        for(const line of lines){
          if(!line.trim()) continue; let payload; try{ payload = JSON.parse(line); }catch{ continue; }
          const msg = payload.message || {};
          if(typeof msg.content === 'string' && msg.content){ assistantBuffer += msg.content; bubble.innerHTML = markedParse(assistantBuffer.trim()); renderMarkdownSafe(bubble); scrollToBottom(); }
          if(payload.done && payload.error){ assistantBuffer += `\n\n*Error: ${payload.error}*`; bubble.innerHTML = markedParse(assistantBuffer.trim()); renderMarkdownSafe(bubble); }
        }
      }
      // Experimental tool commands removed
      try {
        const wrap = bubble.parentElement;
        if(wrap){
          wrap.dataset.final = '1';
          wrap.dataset.regenerating = '';
          if(window.attachActions) window.attachActions(wrap);
        }
      } catch {}
    }catch(e){ bubble.innerHTML = `<div style="color:#ffb4b4">${escapeHTML(e?.message||'Stream error')}</div>`; }
    finally{ isStreaming = false; currentSID = null; streamCtrl = null; }
  }

  // --- Models UI ---
  function updateModelHeader(){
    if(!els.modelName) return;
    const friendly = !!useFriendlyNames;
    const label = currentModel ? ((friendly ? (modelMeta[currentModel.id]?.label) : null) || currentModel.id) : 'Models';
    const provider = currentModel ? (currentModel.provider === 'ollama' ? 'O' : 'L') : '';
    els.modelName.innerHTML = `${escapeHTML(label)} <span class="provider-badge">${provider}</span>`;
  }
  function renderModelsPopup(){
    if(!els.modelPop) return;
    const friendly = !!useFriendlyNames;
    const items = models.map(m=>{
      const label = (friendly ? (modelMeta[m.id]?.label) : null) || m.id;
      const desc = modelMeta[m.id]?.description || '';
      const active = (m.id===currentModel?.id && m.provider===currentModel?.provider)?' active':'';
      const provider = m.provider === 'ollama' ? 'O' : 'L';
      return `<div class="model-item${active}" data-model-id="${escapeHTML(m.id)}" data-provider="${escapeHTML(m.provider)}">
        <div class="model-info">
          <div class="model-name">${escapeHTML(label)} <span class="provider-badge">${provider}</span></div>
          <div class="model-description">${escapeHTML(desc)}</div>
        </div>
        ${active?'<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 6L9 17L4 12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>':''}
      </div>`;
    }).join('');
    const controls = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px;border-top:1px solid var(--border-color)">
        <div style="display:flex;align-items:center;gap:6px">
          <span style="font-size:12px;color:var(--text-secondary)">Names:</span>
          <div role="group" aria-label="Name style" style="display:flex;border:1px solid var(--border-color);border-radius:999px;overflow:hidden">
            <button type="button" data-name-style="dev" style="padding:4px 10px;background:${!friendly?'#3a3a3a':'transparent'};color:#fff;border:none">Dev</button>
            <button type="button" data-name-style="ai" style="padding:4px 10px;background:${friendly?'#3a3a3a':'transparent'};color:#fff;border:none">AI</button>
          </div>
        </div>
        <button type="button" data-regenerate-names style="padding:6px 10px;border:1px solid var(--border-color);border-radius:8px;background:transparent;color:var(--text-primary)">Regenerate</button>
      </div>`;
    els.modelPop.innerHTML = (items || '<div style="padding:8px;color:#9db0c8">No models</div>') + controls;
    qsa('.model-item', els.modelPop).forEach(it=>{
      it.addEventListener('click', (e)=>{
        e.stopPropagation();
        const modelId = it.getAttribute('data-model-id');
        const provider = it.getAttribute('data-provider');
        if(!modelId || !provider) return;
        currentModel = {id: modelId, provider: provider};
        updateModelHeader();
        els.modelPop.classList.remove('visible');
        renderModelsPopup();
      });
    });

    // Toggle name style
    const devBtn = els.modelPop.querySelector('[data-name-style="dev"]');
    const aiBtn  = els.modelPop.querySelector('[data-name-style="ai"]');
    if(devBtn) devBtn.addEventListener('click', (e)=>{ e.stopPropagation(); useFriendlyNames = false; localStorage.setItem('useFriendlyModelNames','0'); updateModelHeader(); renderModelsPopup(); });
    if(aiBtn) aiBtn.addEventListener('click', (e)=>{ e.stopPropagation(); useFriendlyNames = true; localStorage.setItem('useFriendlyModelNames','1'); updateModelHeader(); renderModelsPopup(); });

    // Regenerate names button
    const regenBtn = els.modelPop.querySelector('[data-regenerate-names]');
    if(regenBtn){
      regenBtn.addEventListener('click', async (e)=>{
        e.stopPropagation();
        if(regenBtn.dataset.busy==='1') return;
        try{
          regenBtn.dataset.busy='1';
          const prev = regenBtn.textContent;
          regenBtn.textContent = 'Regeneratingâ€¦';
          regenBtn.style.opacity = '0.7';
          const r = await fetch('/api/model-names/generate', { method:'POST' });
          if(!r.ok){
            const t = await r.text();
            throw new Error(t || ('HTTP '+r.status));
          }
          const data = await r.json();
          modelMeta = data?.meta || modelMeta;
          updateModelHeader();
          renderModelsPopup();
        }catch(err){
          try{ alert('Failed to regenerate names. Ensure Ollama is running.'); }catch{}
          console.error('regenerate names failed', err);
        }finally{
          regenBtn.dataset.busy='';
          try{ regenBtn.textContent = 'Regenerate'; regenBtn.style.opacity='1'; }catch{}
        }
      });
    }
  }

  function setupModelPopup(){
    if(!els.modelBtn || !els.modelPop) return;
    els.modelBtn.addEventListener('click', (e)=>{
      e.stopPropagation();
      els.modelPop.classList.toggle('visible');
    });
    document.addEventListener('click', ()=>{
      els.modelPop.classList.remove('visible');
    });
  }

  // --- Sidebar Chats ---
  function renderChatsList(chats){
    try{
      const wrapper = document.createElement('div');
      wrapper.className = 'sidebar-section';
      wrapper.innerHTML = `<h3 class="section-title">CHATS</h3>` + (chats.map(c=>`<a href="#" class="nav-item" data-id="${escapeHTML(c.id)}" style="display:flex;align-items:center;justify-content:space-between;gap:8px">
        <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHTML(c.title||'Chat')}</span>
        <button type="button" class="input-button" data-more="${escapeHTML(c.id)}" title="More" style="width:28px;height:28px;display:flex;align-items:center;justify-content:center;color:var(--text-secondary)">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 13C12.5523 13 13 12.5523 13 12C13 11.4477 12.5523 11 12 11C11.4477 11 11 11.4477 11 12C11 12.5523 11.4477 13 12 13Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M19 13C19.5523 13 20 12.5523 20 12C20 11.4477 19.5523 11 19 11C18.4477 11 18 11.4477 18 12C18 12.5523 18.4477 13 19 13Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M5 13C5.55228 13 6 12.5523 6 12C6 11.4477 5.55228 11 5 11C4.44772 11 4 11.4477 4 12C4 12.5523 4.44772 13 5 13Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
      </a>`).join(''));
      const mid = els.sidebarMiddle;
      if(mid){ mid.innerHTML = ''; mid.appendChild(wrapper); }
      qsa('a.nav-item[data-id]', wrapper).forEach(a=>{
        a.addEventListener('click', (e)=>{ if(e.target.closest('button[data-more]')) return; e.preventDefault(); loadChat(a.getAttribute('data-id')); });
      });
      // 3-dots menu for delete/rename
      qsa('button[data-more]', wrapper).forEach(btn=>{
        btn.addEventListener('click', (e)=>{
          e.preventDefault(); e.stopPropagation();
          const id = btn.getAttribute('data-more');
          const menu = document.createElement('div');
          Object.assign(menu.style, { position:'fixed', background:'var(--popup-bg)', border:'1px solid var(--border-color)', borderRadius:'8px', padding:'6px', zIndex: 3000, boxShadow:'0 8px 16px rgba(0,0,0,0.3)' });
          menu.innerHTML = `<div data-act="rename" style="padding:8px 10px;cursor:pointer;border-radius:6px">Rename</div><div data-act="delete" style="padding:8px 10px;cursor:pointer;border-radius:6px;color:#ffb4b4">Delete</div>`;
          document.body.appendChild(menu);
          const r = btn.getBoundingClientRect();
          menu.style.left = Math.min(window.innerWidth-10, r.right+6)+'px';
          menu.style.top = Math.max(6, r.top)+'px';
          const cleanup = ()=>{ document.removeEventListener('mousedown', onDoc); try{menu.remove();}catch{} };
          const onDoc = (ev)=>{ if(!menu.contains(ev.target)) cleanup(); };
          document.addEventListener('mousedown', onDoc);
          qsa('[data-act]', menu).forEach(it=> it.addEventListener('click', async ()=>{
            const act = it.getAttribute('data-act');
            cleanup();
            if(act==='delete'){
              if(confirm('Delete this chat?')){ await fetch(`/api/chats/${id}`, { method:'DELETE' }); await refreshChats(); if(currentChatId===id){ currentChatId=null; els.chatContent.innerHTML=''; setChatEmpty(true); updateComposerVisibility(true);} }
            } else if(act==='rename'){
              const title = prompt('New title:'); if(title && title.trim()){ await fetch(`/api/chats/${id}/rename`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ title: title.trim() }) }); await refreshChats(); }
            }
          }));
        });
      });
    }catch(e){ console.error('renderChatsList failed', e); }
  }

  // --- Settings ---
  function openRenameModelsDialog(){
    const dlg = document.createElement('div');
    Object.assign(dlg.style, { position:'fixed', inset:'0', background:'rgba(0,0,0,0.4)', display:'flex', alignItems:'center', justifyContent:'center', zIndex:2000 });
    dlg.innerHTML = `<div style="background:var(--modal-bg);border:1px solid var(--border-color);border-radius:12px;padding:14px;min-width:520px;max-width:90vw">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px"><h3 style="margin:0">Edit Model Names</h3><button id="closeRenameDlg" class="btn-pill">Close</button></div>
      <div style="max-height:60vh;overflow:auto">
        ${models.map(m=>{
          const id = m.id; const label = (modelMeta[id]?.label)||id; const prov = m.provider;
          return `<div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border-color)">
            <div style="flex:0 0 26px"><span class="provider-badge">${prov==='ollama'?'O':'L'}</span></div>
            <div style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--text-secondary)">${id}</div>
            <input data-mid="${id}" type="text" value="${label.replace(/"/g,'&quot;')}" style="flex:0 0 240px;background:var(--main-bg);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);padding:6px 8px">
            <button data-save="${id}" class="btn-pill">Save</button>
          </div>`;
        }).join('')}
      </div>
    </div>`;
    document.body.appendChild(dlg);
    dlg.querySelector('#closeRenameDlg')?.addEventListener('click', ()=> dlg.remove());
    qsa('button[data-save]', dlg).forEach(btn=> btn.addEventListener('click', async ()=>{
      const mid = btn.getAttribute('data-save'); const inp = dlg.querySelector(`input[data-mid="${CSS.escape(mid)}"]`); const label = (inp?.value||'').trim(); if(!label){ toast('Enter a name'); return; }
      try{
        const r = await fetch('/api/model-names', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ id: mid, label }) });
        if(!r.ok){ toast('Save failed'); return; }
        const j = await r.json(); window.modelMeta = j.meta || modelMeta; await loadModels(); toast('Saved');
      }catch{ toast('Save failed'); }
    }));
  }
  function buildSettingsNav(){
    // Use the dedicated nav container; if absent, create it inside the sidebar
    let parent = qs('.settings-sidebar .settings-nav');
    const sidebar = qs('.settings-sidebar');
    if(!parent && sidebar){ parent = document.createElement('div'); parent.className = 'settings-nav'; sidebar.appendChild(parent); }
    if(!parent) return;
    const items = [
      ['general', 'General', '<path d="M14 6.13914C14 6.63933 14.3971 7.04289 14.892 7.04289H15.892C16.892 7.04289 17.581 7.04289 18.06 7.2869C18.472 7.49692 18.7999 7.82484 19.0099 8.23681C19.2539 8.71585 19.2539 9.40483 19.2539 10.4048V13.5952C19.2539 14.5952 19.2539 15.2842 19.0099 15.7632C18.7999 16.1752 18.472 16.5031 18.06 16.7131C17.581 16.9571 16.892 16.9571 15.892 16.9571H14.892C14.3971 16.9571 14 17.3607 14 17.8609V17.8609C14 18.3376 13.901 18.8053 13.7136 19.2319C13.2841 20.2215 12.443 20.9193 11.4168 21.1685C10.9546 21.2827 10.4724 21.3289 10 21.3289V21.3289C6.80004 21.3289 4.67114 18.6738 4.67114 15.4289V8.57114C4.67114 5.32618 6.80004 2.67114 10 2.67114V2.67114C10.4724 2.67114 10.9546 2.71732 11.4168 2.83155C12.443 3.08074 13.2841 3.77854 13.7136 4.76813C13.901 5.19472 14 5.66239 14 6.13914V6.13914Z"/>'],
      ['models', 'Models', '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>'],
      ['language', 'Language', '<circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 0 20"/><path d="M12 2a15.3 15.3 0 0 0 0 20"/>'],
      ['context', 'Context', '<path d="M4 14.5A8.5 8.5 0 1 0 12 6"/><path d="M12 12v-2"/><circle cx="12" cy="12" r="10"/>' ],
      ['ui', 'Interface', '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline>'],
      
    ];
    parent.innerHTML = items.map(([key,label,icon],i)=>`<a class="settings-nav-item${i===0?' active':''}" data-section="${key}"><svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" stroke="currentColor" stroke-width="2">${icon}</svg>${label}</a>`).join('');
    els.settingsSidebar = parent;
  }

  function renderSettingsSection(key){
    const body = qs('.settings-content'); if(!body) return; els.settingsBody = body; body.innerHTML='';
    if(key==='general'){
      body.innerHTML = `
        <h2>General</h2>
        <div class="setting-row"><div><div class="setting-label">Display name</div><div class="setting-description">Name shown in the sidebar.</div></div><div class="setting-control"><input id="setDisplayNameNU" type="text" style="background:var(--main-bg);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);padding:6px 10px;min-width:200px"></div></div>
        <div class="setting-row"><div class="setting-label">Interface animations</div><div class="setting-control"><label class="toggle-switch"><input id="setAnimationNU" type="checkbox"><span class="slider"></span></label></div></div>
        <div class="setting-row"><div class="setting-label">Loading animation</div><div class="setting-control"><button id="chooseLoaderNU" class="btn-pill">Choose…</button><div id="currentLoaderNU" style="color:var(--text-secondary);font-size:12px;margin-left:6px"></div></div></div>
        <div class="setting-row"><div class="setting-label">Disable system prompt (developer)</div><div class="setting-control"><label class="toggle-switch"><input id="setDisableSysPromptNU" type="checkbox"><span class="slider"></span></label></div></div>
      `;
      // values
      const nameEl = byId('setDisplayNameNU'); if(nameEl) nameEl.value = (qs('.user-name')?.textContent||'');
      const animEl = byId('setAnimationNU'); if(animEl) animEl.checked = !!(currentSettings?.ui?.animation);
      if(nameEl){ nameEl.addEventListener('keydown', async (e)=>{ if(e.key==='Enter'){ try{ await fetch('/api/user',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:nameEl.value.trim()||'User'})}); await loadUser(); toast('Saved name'); }catch{} } }); }
      if(animEl){ animEl.addEventListener('change', ()=>{ const nextUI = Object.assign({}, currentSettings?.ui||{}, { animation: !!animEl.checked }); saveSettings({ ui: nextUI }); }); }
      // Loader picker wiring
      const curL = byId('currentLoaderNU'); if(curL){ try{ curL.textContent = (localStorage.getItem('loaderAnimationLabel')||'Default'); }catch{} }
      const btnL = byId('chooseLoaderNU'); if(btnL){ btnL.addEventListener('click', async ()=>{ try{ if(window.LoaderPicker){ await window.LoaderPicker.buildCatalog(); await window.LoaderPicker.openPicker(); if(curL) curL.textContent = (localStorage.getItem('loaderAnimationLabel')||'Default'); } }catch(e){ console.error('loader picker failed', e); } }); }
    } else if(key==='models'){
      const list = models.map(m=>`<div class="dropdown-option" data-model-id="${escapeHTML(m.id)}" data-provider="${escapeHTML(m.provider)}">${escapeHTML(modelMeta[m.id]?.label||m.id)} <span class="provider-badge">${m.provider === 'ollama' ? 'O' : 'L'}</span></div>`).join('');
      body.innerHTML = `
        <h2>Models</h2>
        <div class="setting-row"><div class="setting-label">Default model</div><div class="setting-control"><div class="dropdown-imitation" id="defModelTrigger"><span id="defModelDisplay">${escapeHTML(modelMeta[currentSettings?.default_model?.id||'']?.label||currentSettings?.default_model?.id||currentModel?.id||'Select')}</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 9L12 15L18 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="settings-dropdown-popup" id="defModelPopup">${list}</div></div></div>
        <div class="setting-row"><div class="setting-label">Default provider</div><div class="setting-control"><div class="dropdown-imitation" id="defProvTrigger"><span id="defProvDisplay">${currentSettings?.default_provider || 'ollama'}</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 9L12 15L18 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="settings-dropdown-popup" id="defProvPopup"><div class="dropdown-option" data-value="ollama">ollama</div><div class="dropdown-option" data-value="llamacpp">llamacpp</div></div></div></div>
        <div class="setting-row"><div><div class="setting-label">Rename models</div><div class="setting-description">Customize display names shown in pickers.</div></div><div class="setting-control"><button id="openRenameModels" class="btn-pill">Edit Names</button> <a class="btn-pill" href="/model-store.html" target="_blank">Open Model Store</a></div></div>
        <div class="setting-row"><div><div class="setting-label">Llama.cpp</div><div class="setting-description">Download and set up the llama.cpp server binary.</div></div><div class="setting-control"><button id="downloadLlamaCpp" class="btn-pill">Download llama.cpp</button></div></div>
        <div class="setting-row"><div><div class="setting-label">Llama.cpp GPU Layers</div><div class="setting-description">Number of layers to offload to GPU. Requires restart.</div></div><div class="setting-control"><input id="setLlamaGpuLayers" type="number" min="0" style="width:80px;background:var(--main-bg);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);padding:6px 10px;"></div></div>
        <div class="setting-row"><div><div class="setting-label">Llama.cpp Server Path</div><div class="setting-description">Full path to llama-server.exe (e.g., C:\llama.cpp\llama-server.exe).</div></div><div class="setting-control"><input id="setLlamaCppServerPathNU" type="text" style="width:100%;background:var(--main-bg);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);padding:6px 10px;"></div></div>
        <div class="setting-row"><div><div class="setting-label">Llama.cpp Model ID</div><div class="setting-description">Hugging Face model ID or local .gguf filename (e.g., TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf).</div></div><div class="setting-control"><input id="setLlamaCppModelIdNU" type="text" style="width:100%;background:var(--main-bg);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);padding:6px 10px;"></div></div>
        <div class="setting-row"><div><div class="setting-label">Hugging Face Token</div><div class="setting-description">Used to access gated models (e.g., Gemma). Stored locally. Hidden unless shown.</div></div><div class="setting-control" style="display:flex;gap:8px;align-items:center;min-width:320px"><input id="setHfTokenNU" type="password" placeholder="hf_xxx..." autocomplete="off" style="flex:1;min-width:240px;background:var(--main-bg);border:1px solid var(--border-color);border-radius:8px;color:var(--text-primary);padding:6px 10px;"><button id="toggleHfTokenNU" class="btn-pill" type="button">Show</button><button id="saveHfTokenNU" class="btn-pill" type="button">Save</button><button id="clearHfTokenNU" class="btn-pill" type="button">Clear</button></div></div>
      `;
      const trig = byId('defModelTrigger'); const pop = byId('defModelPopup');
      if(trig){ trig.addEventListener('click', (e)=>{ e.stopPropagation(); pop.classList.toggle('visible'); }); }
      if(pop){ qsa('.dropdown-option', pop).forEach(opt=> opt.addEventListener('click', ()=>{ const modelId = opt.getAttribute('data-model-id'); const provider = opt.getAttribute('data-provider'); byId('defModelDisplay').textContent = modelMeta[modelId]?.label||modelId; saveSettings({ default_model: {id: modelId, provider: provider} }); pop.classList.remove('visible'); })); }
      document.addEventListener('click', ()=>{ try{ pop.classList.remove('visible'); }catch{} });

      const provTrig = byId('defProvTrigger'); const provPop = byId('defProvPopup');
      if(provTrig){ provTrig.addEventListener('click', (e)=>{ e.stopPropagation(); provPop.classList.toggle('visible'); }); }
      if(provPop){ qsa('.dropdown-option', provPop).forEach(opt=> opt.addEventListener('click', ()=>{ const v = opt.getAttribute('data-value'); byId('defProvDisplay').textContent = v; saveSettings({ default_provider: v }); provPop.classList.remove('visible'); })); }
      document.addEventListener('click', ()=>{ try{ provPop.classList.remove('visible'); }catch{} });

      const gpuLayers = byId('setLlamaGpuLayers');
      const llamaCppServerPath = byId('setLlamaCppServerPathNU');
      const llamaCppModelId = byId('setLlamaCppModelIdNU');
      const hfTokenInput = byId('setHfTokenNU');
      const hfTokenToggle = byId('toggleHfTokenNU');
      const hfTokenClear = byId('clearHfTokenNU');
      const hfTokenSave = byId('saveHfTokenNU');

      if(gpuLayers) {
        gpuLayers.value = currentSettings?.llamacpp_gpu_layers || 0;
        gpuLayers.addEventListener('change', () => {
            saveSettings({ llamacpp_gpu_layers: parseInt(gpuLayers.value, 10) || 0 });
        });
      }
      if(llamaCppServerPath) {
        llamaCppServerPath.value = currentSettings?.llamacpp_server_path || '';
        llamaCppServerPath.addEventListener('change', () => {
            saveSettings({ llamacpp_server_path: llamaCppServerPath.value.trim() || null });
        });
        // Auto-detect if empty
        if(!llamaCppServerPath.value){
          try {
            fetch('/api/llamacpp/auto-detect', { method:'POST' })
              .then(r=>r.json().catch(()=>({})))
              .then(j=>{
                if(j && j.ok && j.path){
                  llamaCppServerPath.value = j.path;
                  saveSettings({ llamacpp_server_path: j.path });
                  toast('Detected llama.cpp server');
                }
              }).catch(()=>{});
          }catch{}
        }
      }
      if(llamaCppModelId) {
        llamaCppModelId.value = currentSettings?.llamacpp_model_id || '';
        llamaCppModelId.addEventListener('change', () => {
            saveSettings({ llamacpp_model_id: llamaCppModelId.value.trim() || null });
        });
      }
      if(hfTokenInput){
        try{ hfTokenInput.value = currentSettings?.huggingface_token ? String(currentSettings.huggingface_token) : ''; }catch{}
        const commit = ()=>{
          const val = hfTokenInput.value.trim();
          saveSettings({ huggingface_token: val || null });
          toast(val ? 'Saved Hugging Face token' : 'Cleared Hugging Face token');
        };
        hfTokenInput.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ commit(); } });
        hfTokenInput.addEventListener('blur', ()=>{ /* do not auto-save on blur to avoid accidental clears */ });
        if(hfTokenToggle){
          hfTokenToggle.addEventListener('click', ()=>{
            const isPwd = hfTokenInput.type === 'password';
            hfTokenInput.type = isPwd ? 'text' : 'password';
            hfTokenToggle.textContent = isPwd ? 'Hide' : 'Show';
          });
        }
        if(hfTokenSave){ hfTokenSave.addEventListener('click', commit); }
        if(hfTokenClear){
          hfTokenClear.addEventListener('click', ()=>{ hfTokenInput.value=''; commit(); });
        }
      }
      if(byId('openRenameModels')){
        byId('openRenameModels').addEventListener('click', ()=>{
          openRenameModelsDialog();
        });
      }
      // Starter Packs button removed from settings; available in Model Store
      if(byId('downloadLlamaCpp')){
        byId('downloadLlamaCpp').addEventListener('click', async ()=>{
          try{
            const r = await fetch('/api/install/llamacpp', { method:'POST' });
            const j = await r.json();
            if(j.ok){
              if(j.path){
                if(llamaCppServerPath){ llamaCppServerPath.value = j.path; }
                saveSettings({ llamacpp_server_path: j.path });
                toast(j.note || 'llama.cpp ready');
              } else {
                toast(j.note || 'llama.cpp install attempted');
              }
            } else {
              toast(j.error || 'llama.cpp install failed');
            }
          }catch(e){ toast('llama.cpp install failed'); }
        });
      }
    } else if(key==='language'){
      const langs = [ ['en','English'],['es','Spanish'],['fr','French'],['de','German'],['it','Italian'],['pt','Portuguese'],['ru','Russian'],['zh','Chinese'],['ja','Japanese'],['ko','Korean'],['ar','Arabic'],['hi','Hindi'],['tr','Turkish'],['nl','Dutch'],['pl','Polish'],['sv','Swedish'],['da','Danish'],['no','Norwegian'],['fi','Finnish'],['cs','Czech'],['el','Greek'],['th','Thai'] ];
      const cur = currentSettings?.language?.code || 'en';
      body.innerHTML = `
        <h2>Language</h2>
        <div class="setting-row"><div class="setting-label">Preferred language</div><div class="setting-control"><div class="dropdown-imitation" id="langTrigger"><span id="langDisplay"></span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 9L12 15L18 9" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div class="settings-dropdown-popup" id="langPopup">${langs.map(([code,name])=>`<div class="dropdown-option" data-value="${code}">${name}</div>`).join('')}</div></div></div>
      `;
      const disp = byId('langDisplay'); if(disp){ const m = new Map(langs); disp.textContent = m.get(cur)||'English'; }
      const trig = byId('langTrigger'); const pop = byId('langPopup');
      if(trig){ trig.addEventListener('click', (e)=>{ e.stopPropagation(); pop.classList.toggle('visible'); }); }
      if(pop){ qsa('.dropdown-option', pop).forEach(opt=> opt.addEventListener('click', ()=>{ const code=opt.getAttribute('data-value'); const name=opt.textContent.trim(); byId('langDisplay').textContent = name; saveSettings({ language: { code, name } }); pop.classList.remove('visible'); })); }
      document.addEventListener('click', ()=>{ try{ pop.classList.remove('visible'); }catch{} });
    } else if(key==='context'){
      const enabled = !!(currentSettings?.context_meter?.enabled);
      body.innerHTML = `
        <h2>Context</h2>
        <div class="setting-row"><div class="setting-label">Context meter</div><div class="setting-control"><label class="toggle-switch"><input id="setContextMeterNU" type="checkbox" ${enabled?'checked':''}><span class="slider"></span></label></div></div>
        <div class="setting-row"><div class="setting-description">Shows estimated token usage for system, history, and draft.</div></div>
      `;
      const chk = byId('setContextMeterNU'); if(chk){ chk.addEventListener('change', ()=>{ saveSettings({ context_meter: { enabled: !!chk.checked } }); }); }
    } else if (key === 'ui') {
        body.innerHTML = `
        <h2>Interface</h2>
        <div class="setting-row"><div class="setting-label">No additional interface settings</div></div>
      `;
    }
  }

  function openSettings(){ els.settingsOverlay.classList.add('visible'); }
  function closeSettings(){ els.settingsOverlay.classList.remove('visible'); }

  function setupSettings(){
    buildSettingsNav();
    // default section
    renderSettingsSection('general');
    // sidebar clicks
    qsa('.settings-nav-item', qs('.settings-sidebar')).forEach(a=>{
      a.addEventListener('click', ()=>{
        qsa('.settings-nav-item', qs('.settings-sidebar')).forEach(x=>x.classList.remove('active'));
        a.classList.add('active');
        const k = a.getAttribute('data-section');
        renderSettingsSection(k);
      });
    });
    // overlay close
    if(els.settingsCloseBtn){ els.settingsCloseBtn.addEventListener('click', closeSettings); }
    if(els.settingsOverlay){ els.settingsOverlay.addEventListener('mousedown', (e)=>{ if(e.target===els.settingsOverlay) closeSettings(); }); }
  }

  // --- Search ---
  async function ensureSearchData(){
    if(searchData.length) return;
    try{ const r = await fetch('/api/chats/all'); const d = await r.json(); searchData = d.chats||[]; }catch(e){ console.error('search load failed', e); }
  }
  function openSearch(){ if(!els.searchOverlay) return; els.searchOverlay.classList.add('visible'); try{ els.searchInput.focus(); }catch{} renderSearchResults(''); }
  function closeSearch(){ if(!els.searchOverlay) return; els.searchOverlay.classList.remove('visible'); }
  function renderSearchResults(q){
    const query = (q||'').trim().toLowerCase();
    const root = els.searchResults; if(!root) return;
    if(!query){ root.innerHTML = '<div style="color:var(--text-secondary)">Start typingâ€¦</div>'; return; }
    const rows = [];
    for(const c of searchData){
      let snippet='';
      if((c.title||'').toLowerCase().includes(query)) snippet='Title matches';
      if(!snippet){
        for(const m of (c.messages||[])){
          const i = (m.content||'').toLowerCase().indexOf(query);
          if(i!==-1){ snippet = (m.content||'').slice(Math.max(0,i-40), i+query.length+60).replace(/\s+/g,' ').trim(); break; }
        }
      }
      if(snippet) rows.push({ id:c.id, title:c.title||'Chat', snippet });
    }
    if(!rows.length){ root.innerHTML = '<div style="color:var(--text-secondary)">No matches</div>'; return; }
    root.innerHTML = rows.slice(0,200).map(r=>`<div class="nav-item" data-id="${escapeHTML(r.id)}" style="display:flex;justify-content:space-between;align-items:center"><div style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"><div>${escapeHTML(r.title)}</div><div style="color:var(--text-secondary);font-size:12px">${escapeHTML(r.snippet)}</div></div><svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 6l6 6-6 6" stroke="currentColor" stroke-width="2"/></svg></div>`).join('');
    qsa('.nav-item[data-id]', root).forEach(row=> row.addEventListener('click', async ()=>{ const id=row.getAttribute('data-id'); closeSearch(); await loadChat(id); }));
  }

  // --- Attachments ---
  function ensureDraftId(){ if(!currentDraftId) currentDraftId = (crypto?.randomUUID ? crypto.randomUUID() : 'draft_'+Math.random().toString(36).slice(2)); return currentDraftId; }
  async function handleFiles(files){
    if(!files || !files.length) return;
    if(!currentChatId){ try{ const r = await fetch('/api/chats', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ title: 'New chat with files' })}); const data = await r.json(); currentChatId = data.id; await refreshChats(); }catch(e){ toast('Could not create a new chat for attachments.'); return; } }
    ensureDraftId();
    for(const file of files){
      const ext = ('.'+(file.name.split('.').pop()||'')).toLowerCase();
      if(!ALLOWED_EXTS.has(ext)){ toast(`Unsupported type: ${file.name}`); continue; }
      const text = await file.text();
      const tokens = estimateTokens(text);
      if(tokens > PER_FILE_TOKEN_LIMIT){ toast(`${file.name} exceeds ${PER_FILE_TOKEN_LIMIT} tokens`); continue; }
      if(currentAttachmentsTokens + tokens > TOTAL_TOKEN_LIMIT){ toast(`Adding ${file.name} exceeds total ${TOTAL_TOKEN_LIMIT} tokens`); continue; }
      const fd = new FormData();
      fd.append('file', new Blob([text], { type:'text/plain' }), file.name);
      try{
        const res = await fetch(`/api/chats/${currentChatId}/drafts/${currentDraftId}/attachments`, { method:'POST', body: fd });
        const data = await res.json();
        if(!res.ok || !data.ok){ toast(data.error || 'Upload failed'); continue; }
        currentAttachmentsTokens = data.total_tokens || (currentAttachmentsTokens + tokens);
        toast(`Attached ${file.name} (${tokens} tok)`);
        await fetchAttachments();
      }catch(e){ console.error('upload failed', e); toast('Upload failed'); }
    }
  }

  // --- Forms ---
  function resizeTextarea(t){ if(!t) return; t.style.height='auto'; t.style.height = `${t.scrollHeight}px`; }
  function wireForm(formEl, inputEl){
    if(!formEl || !inputEl) return;
    const btn = formEl.querySelector('.send-button');
    const toggleBtn = ()=>{ if(btn) btn.classList.toggle('active', !!inputEl.value.trim()); };
    inputEl.addEventListener('input', ()=>{ resizeTextarea(inputEl); toggleBtn(); });
    inputEl.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); formEl.dispatchEvent(new Event('submit')); }});
    formEl.addEventListener('submit', async (e)=>{
      e.preventDefault(); const text = inputEl.value.trim(); if(!text) return;
      if(isShutdown){ toast('Assistant is sleeping. Click to wake.'); return; }
      inputEl.value=''; resizeTextarea(inputEl); toggleBtn();
      setChatEmpty(false);
      await streamChatSend(text);
    });
    toggleBtn(); resizeTextarea(inputEl);
  }

  // --- Toast ---
  function toast(s){ try{ if(window.showToast) window.showToast(s); }catch{} }

  // --- Boot ---
  async function boot(){
    // Replace settings nav with relevant items (remove irrelevant ones)
    try{ buildSettingsNav(); }catch{}

    await loadUser();
    await loadSettings();
    await loadModels();
    await refreshChats();

    setupModelPopup();
    setupSettings();

    // Events
    if(els.newChatBtn) els.newChatBtn.addEventListener('click', newChat);
    if(els.userProfileBtn) els.userProfileBtn.addEventListener('click', openSettings);
    const openSettingsDots = document.getElementById('open-settings-btn');
    if(openSettingsDots) openSettingsDots.addEventListener('click', openSettings);
    if(els.searchBtn) els.searchBtn.addEventListener('click', async (e)=>{ e.preventDefault(); await ensureSearchData(); openSearch(); });
    if(els.searchOverlay) els.searchOverlay.addEventListener('mousedown', (e)=>{ if(e.target===els.searchOverlay) closeSearch(); });
    if(els.searchInput){ els.searchInput.addEventListener('input', async ()=>{ await ensureSearchData(); renderSearchResults(els.searchInput.value||''); }); }

    // Wire forms
    wireForm(els.welcomeForm, els.welcomeInput);
    // If a main footer exists (added below in HTML), wire it as well
    const mainForm = byId('main-chat-form');
    const mainInput = byId('main-chat-input');
    if(mainForm && mainInput) wireForm(mainForm, mainInput);

    // Files via + buttons
    qsa('.add-button').forEach(btn=> btn.addEventListener('click', ()=> els.fileInput?.click()));
    if(els.fileInput){ els.fileInput.addEventListener('change', async ()=>{ const files = Array.from(els.fileInput.files||[]); els.fileInput.value = ''; await handleFiles(files); }); }

    // Initial attachments fetch for any existing draft
    await fetchAttachments();

    // Drag & drop on composer
    bindDragAndDrop();

    // Regenerate integration: listen for custom event from UI
    window.addEventListener('request-regeneration', (e)=>{
      try{
        const id = e.detail?.messageId; if(!id) return;
        const node = document.getElementById(id); if(!node) return;
        // Find nearest previous user message text
        let prev = node.previousElementSibling; let userText = '';
        while(prev){ if(prev.classList?.contains('user-message')){ userText = (prev.querySelector('.message-bubble')?.innerText||'').trim(); break; } prev = prev.previousElementSibling; }
        if(!userText) return;
        const bubble = node.querySelector('.message-bubble'); if(!bubble) return;
        // Start streaming into the placeholder bubble
        streamIntoBubble(userText, bubble);
      }catch(err){ console.error('regenerate failed', err); }
    });

    // Expose minimal API for debugging
    window.NU = { getState: ()=>({ chatId: currentChatId, model: currentModel }), send: streamChatSend };
  }

  // init when DOM ready
  if(document.readyState==='loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
