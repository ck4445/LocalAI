const $=(q,ctx=document)=>ctx.querySelector(q);
const $$=(q,ctx=document)=>[...ctx.querySelectorAll(q)];
const byId=id=>document.getElementById(id);
const escapeHTML=s=>(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// --- Markdown & Syntax Highlighting ---
const safeRenderer = new marked.Renderer();
safeRenderer.html = (html) => escapeHTML(html);
safeRenderer.code = (code, infostring) => {
  const lang = (infostring||'').trim().split(/\s+/)[0]||'';
  let highlighted;
  try {
    highlighted = lang ? hljs.highlight(code, { language: lang }).value : hljs.highlightAuto(code).value;
  } catch {
    highlighted = escapeHTML(code);
  }
  const cls = lang ? `hljs ${escapeHTML(lang)}` : 'hljs';
  return `<pre><code class="${cls}">${highlighted}</code></pre>`;
};
safeRenderer.codespan = (code) => `<code>${escapeHTML(code)}</code>`;
marked.setOptions({
  breaks: true,
  gfm: true,
  renderer: safeRenderer,
  highlight: (code, lang) => {
    try { return hljs.highlight(code, { language: lang }).value } catch { return hljs.highlightAuto(code).value }
  }
});

function renderMarkdown(el){
  $$('pre code',el).forEach(b=>hljs.highlightElement(b));
  try{
    renderMathInElement(el,{
      delimiters:[
        {left:"$$",right:"$$",display:true},
        {left:"$",right:"$",display:false},
        {left:"\\KATEX_INLINE_OPEN",right:"\\KATEX_INLINE_CLOSE",display:false}
      ],
      ignoredTags:["script","noscript","style","textarea","pre","code"]
    })
  }catch(e){console.error("Math rendering failed:",e)}
}

function estimateTokens(t){
  t=t||'';
  return Math.max(1, ((t.length||0)+3) >> 2)
}

// --- UI Utilities ---
function showToast(text){
  let w=document.querySelector('.toast-wrap');
  if(!w){
    w=document.createElement('div');
    w.className='toast-wrap';
    document.body.appendChild(w)
  }
  const t=document.createElement('div');
  t.className='toast';
  t.textContent=text;
  w.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';t.style.transition='opacity .4s ease'},2600);
  setTimeout(()=>t.remove(),3100)
}