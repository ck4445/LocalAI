// --- Globals & Constants from HTML ---
const themeToggleBtn=byId('themeToggleBtn'),sunIcon='<span class="icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2m0 18v2m-9-9h2m18 0h2m-6.36-6.36l-1.42-1.42M5.64 18.36l-1.42-1.42m12.72 0l1.42-1.42m-12.72 0l1.42 1.42"/></svg></span>',moonIcon='<span class="icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg></span>',hljsDark=byId('hljs-dark'),hljsLight=byId('hljs-light');
const animationToggleBtn=byId('animationToggleBtn');
const modelsBtn=byId('modelsBtn'),modelsPop=byId('modelsPop'),modelTitle=byId('modelTitle'),home=byId('home'),chat=byId('chat'),thread=byId('thread'),threadInner=byId('threadInner'),homeInput=byId('homeInput'),homeSend=byId('homeSend'),chatInput=byId('chatInput'),chatSend=byId('chatSend'),newChatBtn=byId('newChatBtn'),chatsList=byId('chatsList'),profileBox=byId('profileBox'),userNameEl=byId('userName'),avatarInitial=byId('avatarInitial'),nameEditWrap=byId('nameEditWrap'),nameInput=byId('nameInput'),homeMicBtn=byId('homeMicBtn'),chatMicBtn=byId('chatMicBtn'),memoriesOpen=byId('memoriesOpen'),memOverlay=byId('memOverlay'),memCloseBtn=byId('memCloseBtn'),settingsOverlay=byId('settingsOverlay'),settingsCloseBtn=byId('settingsCloseBtn'),searchBtn=byId('searchChats');
const chatComposer=$('.chat-composer');
const attachmentsBar=byId('attachmentsBar');
let autoScroll=true;

// --- Animation Manager ---
// Prefer the advanced manager from animations.js when available; otherwise fallback.
window.AnimationManager = window.AnimationManager || {
  init: () => {
    const animsEnabled = localStorage.getItem('animationsEnabled') !== 'false';
    if (byId('setAnimation')) byId('setAnimation').checked = animsEnabled;
  },
  toggleAnimation: () => {
    const animsEnabled = localStorage.getItem('animationsEnabled') !== 'false';
    localStorage.setItem('animationsEnabled', !animsEnabled);
    if (byId('setAnimation')) byId('setAnimation').checked = !animsEnabled;
    showToast(`Animations ${!animsEnabled ? 'enabled' : 'disabled'}.`);
    return !animsEnabled ? 'simple' : 'off';
  },
  getLoader: () => {
    const el = document.createElement('div');
    el.className = 'typing-placeholder';
    return el;
  },
  cleanupLoader: () => {}
};
const Anim = window.AnimationManager;

// --- Theme Management ---
function setTheme(theme){const isLight=theme==='light';document.documentElement.dataset.theme=theme;localStorage.setItem('theme',theme);themeToggleBtn.innerHTML=isLight?moonIcon:sunIcon;themeToggleBtn.title=`Switch to ${isLight?'dark':'light'} mode`;hljsDark.disabled=isLight;hljsLight.disabled=!isLight}
function initTheme(){const savedTheme=localStorage.getItem('theme'),prefersDark=window.matchMedia&&window.matchMedia('(prefers-color-scheme: dark)').matches;setTheme(savedTheme||(prefersDark?'dark':'light'))}
themeToggleBtn.addEventListener('click',()=>{const currentTheme=document.documentElement.dataset.theme||'dark';setTheme(currentTheme==='dark'?'light':'dark')});
animationToggleBtn.addEventListener('click',()=>{Anim.toggleAnimation()});
// Initialize animation manager (sets button icon/state when advanced manager is present)
try { Anim.init(); } catch {}

// --- UI State & Navigation ---
function goHome(){home.classList.remove('hidden');chat.classList.add('hidden');homeInput.focus()}
function goChat(){home.classList.add('hidden');chat.classList.remove('hidden');chatInput.focus()}

// --- Chat Thread UI ---
function scrollToBottom(force=!1){if(force||autoScroll)thread.scrollTop=thread.scrollHeight}
new ResizeObserver(()=>thread.style.paddingBottom=chatComposer.offsetHeight+60+'px').observe(chatComposer);
thread.addEventListener('scroll',()=>autoScroll=thread.scrollHeight-thread.scrollTop-thread.clientHeight<60);

function addUserMessage(text){
  const row=document.createElement('div');
  row.className='msg user';
  row.innerHTML=`<div class=\"center\" style=\"display:flex;gap:12px;justify-content:flex-end;align-items:flex-start\"><div class=\"bubble\">${escapeHTML(text)}</div><div class=\"avatar\">🧑</div></div>`;
  threadInner.appendChild(row);
  updateContextMeter();
}

function addAssistantMessage(markdownText,finalize=!1){
  const row=document.createElement('div');
  row.className='msg assistant';
  row.innerHTML=`<div class="center" style="display:flex;gap:12px;align-items:flex-start">
    <div class="avatar">⚙️</div>
    <div class="assistant-card">
      <div class="md-body"></div>
      <div class="msg-tools" style="${finalize?'':'display:none'}">
        <button class="btn-icon" title="Regenerate response" type="button" data-action="regen">
          <span class="icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 019-9 9 9 0 018.66 6"/><path d="M21 3v6h-6"/><path d="M21 12a9 9 0 01-9 9 9 9 0 01-8.66-6"/><path d="M3 21v-6h6"/></svg></span>
        </button>
        <span class="muted" style="align-self:center;font-size:12px">Reload</span>
      </div>
    </div>
  </div>`;
  const body=row.querySelector('.md-body');
  if(markdownText){
    body.innerHTML=marked.parse(markdownText);
    renderMarkdown(body.parentElement)
  }else{
    const loader=Anim.getLoader();
    body.appendChild(loader)
  }
  row.querySelector('[data-action="regen"]').addEventListener('click',()=>{
    try{
      const prevUserMsgEl = row.previousElementSibling;
      if (prevUserMsgEl && prevUserMsgEl.classList.contains('user')) {
        const userText = prevUserMsgEl.querySelector('.bubble').textContent.trim();
        if (isStreaming) { abortStream(); }
        row.remove();
        prevUserMsgEl.remove();
        chatInput.value = userText;
        handleSend(chatInput);
      } else {
        showToast('No previous user message found to regenerate.');
      }
    } catch(e) { console.error('Regen failed', e); }
  });
  threadInner.appendChild(row);
  if(!finalize)scrollToBottom();
  return body
}

function setSendingState(active){
  isStreaming=active;
  const stopHTML='<span class="icon" style="color:white"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"/></svg></span>';
  const sendHTML='<span class="icon" style="color:white"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4 20-7z"/></svg></span>';
  [chatSend,homeSend].forEach(btn=>{
    btn.classList.toggle('btn-stop',active);
    btn.title=active?'Stop':'Send';
    btn.innerHTML=active?stopHTML:sendHTML;
  });
  if(!active) {
    try{ const tools=threadInner.querySelector('.msg.assistant:last-child .msg-tools'); if(tools) tools.style.display='flex'; }catch{}
  }
}

function renderAttachments(){
    if(!attachmentsBar)return;
    attachmentsBar.innerHTML='';
    if(!currentAttachments.length){
        attachmentsBar.style.display='none';
        return;
    }
    attachmentsBar.style.display='block';
    attachmentsBar.innerHTML=`<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">${currentAttachments.map(a=>`<div class="chip" data-att="${a.id}" title="${escapeHTML(a.name)} (attached to next message)"><span class="icon" style="color:#aab4c6"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9z"/><path d="M14 3v6h6"/></svg></span><span class="name" style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHTML(a.name)}</span><span class="muted" style="margin-left:6px;color:#aab4c6">Attached</span><span class="muted" style="margin-left:6px;color:#aab4c6">${a.tokens} tok</span><button type="button" class="chip-x" title="Remove" data-remove="${a.id}">×</button></div>`).join('')}<span class="muted" style="margin-left:auto;color:#9ed0ff">${currentAttachmentsTokens}/${TOTAL_TOKEN_LIMIT} tok</span></div>`;
}

function renderContextBar(el, used, max){
  if(!el) return;
  const pct = Math.min(100, Math.round((used/max)*100));
  el.innerHTML = `<div style="display:flex;align-items:center;gap:8px"><div style="flex:1;height:8px;background:#0c1219;border:1px solid var(--soft-border);border-radius:999px;overflow:hidden"><div style="width:${pct}%;height:100%;background:linear-gradient(90deg,#4f46e5,#06b6d4)"></div></div><div style="min-width:140px;color:#9fb4ff;font-variant-numeric:tabular-nums">Context: ${used}/${max}</div></div>`;
}


// --- Component Logic: Modals, Popups, etc. ---

function showMiniMenu(anchor,items){
  return new Promise(resolve=>{
    const isLight=(document.documentElement.dataset.theme||'dark')==='light';
    const menuBg=isLight?'#ffffff':'#0f1721';
    const menuBorder=isLight?'rgba(0,0,0,0.10)':'rgba(255,255,255,0.08)';
    const menuShadow=isLight?'0 6px 20px rgba(0,0,0,0.12)':'0 6px 20px rgba(0,0,0,0.4)';
    const rowHoverBg=isLight?'#f1f5f9':'#16202b';
    const textColor=isLight?'#1f2937':'#dfe3ea';
    const rowColor=isLight?'#334155':'#cdd3de';

    const menu=document.createElement('div');
    Object.assign(menu.style,{ position:'fixed', background:menuBg, color:textColor, border:`1px solid ${menuBorder}`, borderRadius:'8px', padding:'6px', fontSize:'13px', boxShadow:menuShadow, zIndex:100 });
    items.forEach(txt=>{
      const r=document.createElement('div');
      r.textContent=txt;
      Object.assign(r.style,{padding:'8px 10px',borderRadius:'6px',cursor:'pointer',color:rowColor});
      r.addEventListener('mouseenter',()=>r.style.background=rowHoverBg);
      r.addEventListener('mouseleave',()=>r.style.background='transparent');
      r.addEventListener('click',()=>{cleanup();resolve(txt)});
      menu.appendChild(r);
    });
    document.body.appendChild(menu);
    const rect=anchor.getBoundingClientRect();
    menu.style.left=Math.min(rect.right+6,window.innerWidth-menu.offsetWidth-8)+'px';
    menu.style.top=Math.max(8,rect.top-4)+'px';
    const cleanup=()=>{document.removeEventListener('mousedown',onDoc);menu.remove()};
    const onDoc=e=>{if(!menu.contains(e.target)){cleanup();resolve(null)}};
    setTimeout(()=>document.addEventListener('mousedown',onDoc),0);
  })
}

function positionModelsPop(){const rect=modelsBtn.getBoundingClientRect(),margin=8,w=Math.min(320,window.innerWidth-margin*2);const prevDisplay=modelsPop.style.display,prevVis=modelsPop.style.visibility;modelsPop.style.visibility='hidden';modelsPop.style.display='block';modelsPop.style.width=w+'px';const h=modelsPop.offsetHeight||280,desiredLeft=rect.left+(rect.width/2)-(w/2);let left=Math.max(margin,Math.min(desiredLeft,window.innerWidth-w-margin));const belowTop=rect.bottom+margin,maxTop=window.innerHeight-h-margin;let top=Math.min(belowTop,Math.max(margin,maxTop));if(belowTop>maxTop&&(rect.top-margin)>=h+margin){top=Math.max(margin,rect.top-h-margin)}modelsPop.style.left=left+'px';modelsPop.style.top=top+'px';modelsPop.style.visibility=prevVis||'';modelsPop.style.display=prevDisplay||''}
let _posRaf=0;function schedulePos(){if(_posRaf)return;_posRaf=requestAnimationFrame(()=>{_posRaf=0;if(modelsPop.classList.contains('visible'))positionModelsPop()})}
// Prevent outside-click handler from closing when re-rendering popup content
let _modelsGuard=false;
window.addEventListener('resize',schedulePos);window.addEventListener('scroll',schedulePos,{passive:true});
document.addEventListener('click',e=>{
  try{
    if(_modelsGuard){ _modelsGuard=false; return }
    if(!modelsPop.contains(e.target)&&!modelsBtn.contains(e.target)){
      modelsPop.classList.remove('visible');
      try{ hiddenModelsOpen=false }catch{}
    }
  }catch{}
});

// --- Search Modal ---
let searchOverlay,searchInput,searchResults,searchData=[];
function ensureSearchUI(){if(searchOverlay)return;searchOverlay=document.createElement('div');searchOverlay.className='search-overlay';searchOverlay.innerHTML='<div class="search-modal"><div class="search-head"><input class="search-input" placeholder="Search chats..." /><button class="btn-icon" title="Close"><span class="icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 6l12 12M18 6L6 18"/></svg></span></button></div><div class="search-results"></div></div>';document.body.appendChild(searchOverlay);searchInput=searchOverlay.querySelector('.search-input');searchResults=searchOverlay.querySelector('.search-results');const close=()=>searchOverlay.classList.remove('visible');searchOverlay.querySelector('.btn-icon').addEventListener('click',close);searchOverlay.addEventListener('mousedown',e=>{if(!e.target.closest('.search-modal'))close()});searchOverlay.addEventListener('keydown',e=>{if(e.key==='Escape')close()})}
let searchT;document.addEventListener('input',e=>{if(e.target===searchInput){clearTimeout(searchT);searchT=setTimeout(()=>renderSearchResults(e.target.value),120)}});
document.addEventListener('keydown',e=>{if(e.target===searchInput&&e.key==='Enter')searchResults?.querySelector('.search-row')?.click()});
function renderSearchResults(q){
  try{
    q=(q||'').trim().toLowerCase();
    const items = $$('#chatsList .item');
    const rows = [];
    items.forEach(it=>{
      const id = it.getAttribute('data-id')||'';
      const title = (it.querySelector('.title')?.textContent||'').trim();
      if(!q || title.toLowerCase().includes(q)){
        rows.push({id,title});
      }
    });
    if(!searchResults) return;
    if(rows.length===0){ searchResults.innerHTML = '<div style="padding:10px;color:#9db0c8">No matches</div>'; return; }
    searchResults.innerHTML = rows.map(r=>`<div class="search-row" data-id="${r.id}" style="padding:8px;cursor:pointer;border-bottom:1px solid var(--soft-border)">${escapeHTML(r.title||'(untitled)')}</div>`).join('');
    $$('.search-row', searchResults).forEach(row=>{
      row.addEventListener('click', async ()=>{
        const id=row.getAttribute('data-id');
        try{ if(typeof loadChat==='function') await loadChat(id); }catch{}
        try{ searchOverlay.classList.remove('visible'); }catch{}
      });
    });
  }catch(e){ console.error('renderSearchResults failed', e); }
}

// Lightweight system notice appended to the thread (e.g., memory saved)
function addMemoryNotice(text){
  try{
    const row=document.createElement('div');
    row.className='msg assistant';
    const safe=escapeHTML(text||'');
    row.innerHTML=`<div class="center" style="display:flex;gap:12px;align-items:flex-start">
      <div class="avatar">💾</div>
      <div class="assistant-card" style="padding:10px 12px"><div class="md-body"><div style="color:#9fb4ff">Saved to memory: <em>${safe}</em></div></div></div>
    </div>`;
    threadInner.appendChild(row);
    scrollToBottom(true);
  }catch(e){ console.error('addMemoryNotice failed', e); }
}
function openSearch(){ ensureSearchUI(); searchOverlay.classList.add('visible'); try{ searchInput.focus(); }catch{} renderSearchResults(''); }
try{ searchBtn?.addEventListener('click',openSearch); }catch{}

// --- Memories Modal ---
function openMemories(){memOverlay.classList.add('visible');loadMemories()}
function closeMemories(){memOverlay.classList.remove('visible')}
try{ memoriesOpen.addEventListener('click',openMemories); }catch{}
try{ memCloseBtn.addEventListener('click',closeMemories); }catch{}
// Close when clicking backdrop itself
try{ memOverlay.addEventListener('mousedown',e=>{ if(e.target===memOverlay) closeMemories() }); }catch{}

// --- Settings Modal ---
function openSettingsModal(){
  if(!currentSettings) return;
  settingsOverlay.classList.add('visible');
  const setThemeSel=byId('setTheme'), setAnimChk=byId('setAnimation'), setDisplayName=byId('setDisplayName'), setUserNameAI=byId('setUserNameAI'), setVerbosity=byId('setVerbosity'), setDefaultModel=byId('setDefaultModel'), setPersEnabled=byId('setPersEnabled'), persList=byId('persList'), persPreview=byId('persPreview'), setLanguageSel=byId('setLanguage'), setContextMeter=byId('setContextMeter'), setDisableSysPrompt=byId('setDisableSysPrompt');
  try{ setThemeSel.value = (currentSettings.ui?.theme)|| (document.documentElement.dataset.theme||'dark'); }catch{}
  try{ setAnimChk.checked = !!(currentSettings.ui?.animation); }catch{}
  setDisplayName.value = userNameEl.textContent.trim();
  try{ setUserNameAI.value = (currentSettings.user?.name||''); }catch{}
  try{ setVerbosity.value = (currentSettings.verbosity||'High'); }catch{}
  try{ setDefaultModel.innerHTML = (models||[]).map(m=>`<option value="${escapeHTML(m)}">${escapeHTML(modelMeta[m]?.label||m)}</option>`).join(''); setDefaultModel.value = currentSettings.default_model || ''; }catch{}
  try{
    setPersEnabled.checked = !!(currentSettings.personality?.enabled);
    const selected = new Set((currentSettings.personality?.selected)||[]);
    persList.innerHTML = Object.keys(PERSONALITY_LIBRARY).map(p=>{
      const cls = selected.has(p) ? 'sm-btn chip selected' : 'sm-btn chip';
      return `<button type=\"button\" data-pers=\"${escapeHTML(p)}\" class=\"${cls}\">${escapeHTML(p)}</button>`;
    }).join('');
    const preview = (currentSettings.personality?.selected||[]).slice(0,3).map(p=>`- ${p}: ${PERSONALITY_LIBRARY[p]||''}`).join('\n');
    persPreview.textContent = preview ? `Preview:\n${preview}` : 'No personalities selected.';
  }catch{}
  try{
    const langs = [ ['en','English'],['es','Spanish'],['fr','French'],['de','German'],['it','Italian'],['pt','Portuguese'],['ru','Russian'],['zh','Chinese'],['ja','Japanese'],['ko','Korean'],['ar','Arabic'],['hi','Hindi'],['tr','Turkish'],['nl','Dutch'],['pl','Polish'],['sv','Swedish'],['da','Danish'],['no','Norwegian'],['fi','Finnish'],['cs','Czech'],['el','Greek'],['th','Thai'] ];
    setLanguageSel.innerHTML = langs.map(([code,name])=>`<option value="${code}">${name}</option>`).join('');
    const lc = currentSettings.language?.code||'en'; setLanguageSel.value = lc;
  }catch{}
  try{ setContextMeter.checked = !!(currentSettings.context_meter?.enabled); }catch{}
  try{ if(setDisableSysPrompt) setDisableSysPrompt.checked = !!(currentSettings.dev?.disable_system_prompt); }catch{}
}
function closeSettingsModal(){ settingsOverlay.classList.remove('visible'); }
try{ settingsCloseBtn.addEventListener('click', closeSettingsModal); }catch{}
// Close when clicking backdrop itself
try{ settingsOverlay.addEventListener('mousedown', e=>{ if(e.target===settingsOverlay) closeSettingsModal(); }); }catch{}

// --- Input Handling (Voice, Drag/Drop) ---
function setupVoice(btn,textarea){
    if(!('webkitSpeechRecognition'in window))return btn.style.display='none';
    let rec=null,active=!1;
    btn.addEventListener('click',()=>{
        if(!rec){
            rec=new webkitSpeechRecognition();
            rec.interimResults=!0;rec.continuous=!1;
            rec.onresult=e=>{let final='';for(let i=0;i<e.results.length;i++)if(e.results[i].isFinal)final+=e.results[i][0].transcript;if(final){const start=textarea.selectionStart,end=textarea.selectionEnd,current=textarea.value;textarea.value=current.substring(0,start)+final.trim()+current.substring(end);textarea.selectionStart=textarea.selectionEnd=start+final.trim().length}};
            rec.onend=()=>{active=!1;btn.style.opacity=1}
        }
        active=!active;active?rec.start():rec.stop();btn.style.opacity=active?.6:1
    })
}
function setupDragTarget(textarea){
    const onDrop=e=>{
        e.preventDefault();
        for(const f of Array.from(e.dataTransfer.files||[])){
            if(f.type.startsWith('text/')){
                const reader=new FileReader();
                reader.onload=()=>textarea.value+=(textarea.value?"\n\n":'')+(reader.result||'');
                reader.readAsText(f)
            }
        }
    };
    textarea.addEventListener('dragover',e=>e.preventDefault());
    textarea.addEventListener('drop',onDrop)
}

// --- Models Popup Rendering (used by app.js) ---
function renderModelsList(){
  try{
    const list = Array.isArray(models) ? models.slice() : [];
    const showingHidden = (typeof hiddenModelsOpen!=="undefined") ? hiddenModelsOpen : false;
    const isHidden = (m)=> (typeof hiddenModels!=="undefined" && hiddenModels instanceof Set) ? hiddenModels.has(m) : false;

    const visible = list.filter(m=>!isHidden(m));
    const rows = (showingHidden ? list : visible).map(m=>{
      const meta = (typeof modelMeta!=="undefined" ? (modelMeta[m]||{}) : {});
      const label = (typeof useFriendlyModelNames!=="undefined" && useFriendlyModelNames) ? (meta.label||m) : m;
      const selected = (typeof currentModel!=="undefined" && currentModel===m);
      const hiddenFlag = isHidden(m);
      return `<div class="model-row${selected?' selected':''}" data-model="${escapeHTML(m)}" style="display:flex;align-items:center;gap:8px;cursor:pointer;padding:8px;border-radius:8px">
        <div style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHTML(label)}</div>
        <button type="button" class="sm-btn" data-hide="${escapeHTML(m)}" title="${hiddenFlag?'Unhide':'Hide'}" style="min-width:60px">${hiddenFlag?'Unhide':'Hide'}</button>
      </div>`;
    }).join('') || '<div style="padding:10px;color:#9db0c8">No models available.</div>';

    const toggle = list.length?`<div style="display:flex;align-items:center;justify-content:space-between;padding:8px;border-top:1px solid var(--soft-border);margin-top:6px">
      <div style="color:#9db0c8">${showingHidden?'Showing all':'Hiding hidden models'}</div>
      <button type="button" class="sm-btn" data-toggle-hidden>${showingHidden?'Hide Hidden':'Show Hidden'}</button>
    </div>`:'';

    modelsPop.innerHTML = `<div style="padding:6px">${rows}</div>${toggle}`;

    // Item click selects model
    $$('.model-row', modelsPop).forEach(row=>{
      row.addEventListener('click', (e)=>{
        const btn = e.target.closest('button[data-hide]');
        if(btn) return; // handled below
        const m = row.getAttribute('data-model');
        if(!m) return;
        try{ currentModel = m; }catch{}
        try{
          const meta = (typeof modelMeta!=="undefined" ? (modelMeta[m]||{}) : {});
          const label = (typeof useFriendlyModelNames!=="undefined" && useFriendlyModelNames) ? (meta.label||m) : m;
          if(modelTitle) modelTitle.textContent = label;
        }catch{}
        modelsPop.classList.remove('visible');
      });
    });

    // Hide/Unhide toggle per model
    $$('button[data-hide]', modelsPop).forEach(btn=>{
      btn.addEventListener('click', (e)=>{
        e.stopPropagation(); e.preventDefault(); _modelsGuard=true;
        const m = btn.getAttribute('data-hide');
        if(!m || typeof hiddenModels==="undefined") return;
        if(hiddenModels.has(m)) hiddenModels.delete(m); else hiddenModels.add(m);
        try{ saveHiddenModels&&saveHiddenModels(); }catch{}
        setTimeout(()=>{ renderModelsList(); schedulePos(); },0);
      });
    });

    // Toggle showing hidden
    const tbtn = modelsPop.querySelector('[data-toggle-hidden]');
    if(tbtn){
      tbtn.addEventListener('click', (e) => {
        e.stopPropagation(); e.preventDefault(); _modelsGuard=true;
        try{ hiddenModelsOpen = !hiddenModelsOpen; }catch{}
        setTimeout(()=>{ renderModelsList(); schedulePos(); },0);
      });
    }
  }catch(e){ console.error('renderModelsList failed', e); }
}
