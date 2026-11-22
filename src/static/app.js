const $ = (q) => document.querySelector(q);
const $$ = (q) => [...document.querySelectorAll(q)];

const MODE_INFO = {
  fortune: {
    icon: '🔮',
    label: '내 관상 알아보기',
    badge: 'FACE READING',
    title: '내 얼굴에는<br /><span>어떤 관상이 담겨 있을까요?</span>',
    subtitle: '눈, 코, 입, 얼굴형의 특징을 살펴보고 전통 관상에서는 어떻게 풀이하는지 알아봐요.'
  },
  animal: {
    icon: '🐾',
    label: '나랑 닮은 동물 찾기',
    badge: 'ANIMAL MATCH',
    title: '나랑 가장 닮은<br /><span>동물을 찾아볼게요</span>',
    subtitle: '얼굴형과 눈매, 전체적인 인상을 분석해 나와 가장 닮은 동물 TOP 3를 찾아봐요.'
  },
  pokemon: {
    icon: '🎮',
    label: '나랑 닮은 포켓몬 찾기',
    badge: 'POKÉMON MATCH',
    title: '나랑 가장 닮은<br /><span>포켓몬을 찾아볼게요</span>',
    subtitle: '얼굴의 분위기와 동물 분석 결과를 바탕으로 나와 닮은 포켓몬을 찾아봐요.'
  },
  all: {
    icon: '✨',
    label: '내 얼굴 종합 분석',
    badge: 'ALL IN ONE',
    title: '한 번의 촬영으로<br /><span>내 얼굴을 모두 알아볼게요</span>',
    subtitle: '관상부터 동물, 포켓몬까지 한 번에 모두 확인해보세요.'
  }
};

const state = {
  mode: null,
  lastCapture: null,
  loadingResult: false,
  result: null
};

const homeSection = $('#homeSection');
const scanFlow = $('#scanFlow');
const resultZone = $('#resultZone');
const cameraState = $('#cameraState');
const startBtn = $('#startBtn');
const cancelBtn = $('#cancelBtn');
const newScanBtn = $('#newScanBtn');
const homeBtn = $('#homeBtn');
const progressBar = $('#progressBar');
const progressText = $('#progressText');
const progressFloating = $('#progressFloating');
const statusText = $('#statusText');
const statusFloating = $('#statusFloating');
const toast = $('#toast');

function showToast(message, error=false){
  toast.textContent = message;
  toast.className = `toast show${error ? ' error' : ''}`;
  setTimeout(() => toast.className = 'toast', 2800);
}

async function jsonFetch(url, options={}){
  const res = await fetch(url, options);
  const data = await res.json();
  if(!res.ok) throw new Error(data.error || 'Request failed');
  return data;
}

function escapeHtml(value){
  return String(value ?? '')
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'",'&#039;');
}

function resultApiUrl(){
  return window.location.pathname === '/preview-result' ? '/api/latest-result' : '/api/result';
}

function selectMode(mode){
  if(!MODE_INFO[mode]) return;
  state.mode = mode;
  const info = MODE_INFO[mode];

  homeSection.hidden = true;
  scanFlow.hidden = false;
  resultZone.hidden = true;
  homeBtn.hidden = false;

  $('#selectedModePill').textContent = `${info.icon} ${info.label}`;
  $('#scanBadge').textContent = info.badge;
  $('#scanTitle').innerHTML = info.title;
  $('#scanSubtitle').textContent = info.subtitle;
  $('#startBtnText').textContent = `${info.icon} 시작하기`;

  window.scrollTo({top:0, behavior:'smooth'});
}

function goHome(){
  homeSection.hidden = false;
  scanFlow.hidden = true;
  resultZone.hidden = true;
  homeBtn.hidden = true;
  state.mode = null;
  window.scrollTo({top:0, behavior:'smooth'});
}

$$('[data-mode]').forEach(btn => {
  btn.addEventListener('click', () => selectMode(btn.dataset.mode));
});

$('#brandHomeBtn').addEventListener('click', goHome);
homeBtn.addEventListener('click', goHome);
$('#backToModesBtn').addEventListener('click', goHome);

function setChecks(checks={}) {
  // Quality checks are kept internal; consumer UI shows only the guide state.
}

function setGuideState(s){
  const wrap = $('#cameraWrap');
  const eyebrow = $('#guideEyebrow');
  if(!wrap) return;

  wrap.classList.remove('guide-warn','guide-ready','guide-collecting','guide-done');

  const raw = s.status || '';
  const checks = s.checks || {};
  let message = '얼굴을 가이드 안에 맞춰주세요';
  let eyebrowText = 'FACE GUIDE';

  if(raw === 'MOVE CLOSER'){
    wrap.classList.add('guide-warn');
    message = '조금 더 가까이 와주세요';
  }else if(raw === 'KEEP FACE INSIDE FRAME'){
    wrap.classList.add('guide-warn');
    message = '얼굴 전체를 가이드 안에 맞춰주세요';
  }else if(raw === 'LOOK STRAIGHT AT CAMERA'){
    wrap.classList.add('guide-warn');
    message = '카메라를 정면으로 바라봐주세요';
  }else if(raw === 'LOOK AT THE CAMERA LENS'){
    wrap.classList.add('guide-warn');
    message = '카메라 렌즈를 바라봐주세요';
  }else if(raw === 'BLINK - HOLD ON'){
    wrap.classList.add('guide-warn');
    message = '눈을 뜨고 잠시 그대로 있어주세요';
  }else if(raw === 'COLLECTING'){
    wrap.classList.add('guide-collecting');
    eyebrowText = 'ANALYZING';
    message = '좋아요! 잠시 그대로 있어주세요';
  }else if(raw === 'READY - AUTO SAVING' || raw === 'SAVING'){
    wrap.classList.add('guide-ready');
    eyebrowText = 'ALMOST DONE';
    message = '분석을 마무리하고 있어요';
  }else if(raw === 'DONE'){
    wrap.classList.add('guide-done');
    eyebrowText = 'COMPLETE';
    message = '분석이 완료됐어요';
  }else if(checks.face && checks.pose && checks.gaze){
    wrap.classList.add('guide-ready');
    message = s.scan_active ? '좋아요! 잠시 그대로 있어주세요' : '좋아요! 분석을 시작해보세요';
  }

  if(eyebrow) eyebrow.textContent = eyebrowText;
  statusFloating.textContent = message;
  statusText.textContent = message;
}

async function pollStatus(){
  try{
    const s = await jsonFetch('/api/status');
    cameraState.classList.toggle('ready', !!s.camera_ready);
    const stateText = cameraState.querySelector('b');
    if(stateText) stateText.textContent = s.camera_ready ? '카메라 준비 완료' : '카메라 연결 중';

    startBtn.disabled = !s.camera_ready || s.scan_active;
    cancelBtn.hidden = !s.scan_active;

    const progress = s.progress || 0;
    progressBar.style.width = `${progress}%`;
    progressText.textContent = `${progress}%`;
    progressFloating.textContent =
      `${String(s.count || 0).padStart(2,'0')} / ${String(s.target || 30).padStart(2,'0')}`;

    setGuideState(s);
    setChecks(s.checks);

    if(s.error) showToast(s.error, true);

    if(s.capture_id && s.capture_id !== state.lastCapture && !state.loadingResult){
      state.lastCapture = s.capture_id;
      await loadResult();
    }
  }catch(err){
    cameraState.classList.remove('ready');
    const stateText = cameraState.querySelector('b');
    if(stateText) stateText.textContent = '서버 연결 대기';
  }
}

startBtn.addEventListener('click', async () => {
  try{
    const data = await jsonFetch('/api/scan/start', {method:'POST'});
    if(!data.ok) throw new Error(data.error || '분석을 시작할 수 없습니다.');
    state.lastCapture = null;
    state.result = null;
    resultZone.hidden = true;
    showToast('분석을 시작할게요. 카메라를 정면으로 바라봐 주세요 ✨');
  }catch(err){
    showToast(err.message, true);
  }
});

cancelBtn.addEventListener('click', async () => {
  await jsonFetch('/api/scan/cancel', {method:'POST'});
  showToast('분석을 취소했어요.');
});

newScanBtn.addEventListener('click', () => {
  resultZone.hidden = true;
  scanFlow.hidden = false;
  $('#scanSection').scrollIntoView({behavior:'smooth', block:'center'});
});

function renderMetrics(metrics){
  $('#metricsGrid').innerHTML = metrics.map(m => {
    const level = Number(m.level);
    const of = Number(m.of || 10);
    const width = Math.max(0, Math.min(100, (level / Math.max(of, 1)) * 100));
    return `
    <div class="metric">
      <div class="metric-head">
        <b>${escapeHtml(m.label)}</b>
        <span class="metric-level">${level}/${of}</span>
      </div>
      <div class="metric-track"><i style="width:${width}%"></i></div>
      <p>${escapeHtml(m.descriptor)}${m.percentile_band ? ` · ${escapeHtml(m.percentile_band)}` : ''}</p>
    </div>`;
  }).join('');
}

const ANIMAL_KO = {
  bear:'곰', cat:'고양이', deer:'사슴', dog:'강아지', fox:'여우',
  hamster:'햄스터', hare:'토끼', hedgehog:'고슴도치', horse:'말',
  lion:'사자', panda:'판다', raccoon:'너구리', squirrel:'다람쥐',
  tiger:'호랑이', wolf:'늑대'
};

function animalKorean(name){
  const key = String(name || '').toLowerCase();
  return ANIMAL_KO[key] || String(name || '알 수 없음');
}

function matchScoreText(item){
  if(item.match_score == null || !Number.isFinite(Number(item.match_score)))
    return '닮음도 산정 중';
  const score=Number(item.match_score).toFixed(1);
  return item.score_kind === 'match'
    ? `닮음도 ${score}점`
    : `TOP3 매칭 비중 ${score}%`;
}

function photo(url,alt,kind,additional=''){
  // The server checks which local image files exist; do not fabricate URLs.
  if(!url) return `<div class="photo-placeholder ${kind}">${kind === 'pokemon' ? '🎮' : '🐾'}</div>`;
  return `<img loading="lazy" class="${escapeHtml(additional)}" src="${escapeHtml(url)}" alt="${escapeHtml(alt)}" />`;
}

function topHero(name,subtitle,photoUrl,kind,scoreText){
  return `
    <div class="match-feature">
      <div class="match-feature-art ${kind}">
        ${photo(photoUrl,name,kind,'match-feature-img')}
      </div>
      <div class="match-feature-copy">
        <div class="rank">BEST MATCH</div>
        <h4>${escapeHtml(name)}</h4>
        <p>${escapeHtml(subtitle)}</p>
        <div class="match-score-pill">${escapeHtml(scoreText)}</div>
      </div>
    </div>
  `;
}

function rankingItem(m,i,mode){
  const pokemon = mode === 'pokemon';
  const name = pokemon ? m.pokemon : animalKorean(m.animal)+'상';
  const source = pokemon ? m.pokemon_image_url : m.animal_image_url;
  const scoreText = pokemon ? `기반 동물상: ${animalKorean(m.animal)}` : matchScoreText(m);
  return `<div class="match-item match-thumbnail-row">
      <div class="match-small-photo ${pokemon?'pokemon':'animal'}">
        ${photo(source,name,pokemon?'pokemon':'animal','match-thumbnail-img')}
      </div>
      <div class="match-item-main">
        <b>#${i + 2} ${escapeHtml(name)}</b>
        <small>${escapeHtml(scoreText)}</small>
      </div>
    </div>`;
}

function renderAnimalMatches(top3){
  const top=$('#topMatch');
  const list=$('#matchList');
  $('#matchEyebrow').textContent='ANIMAL MATCH';
  $('#matchTitle').textContent='나랑 가장 닮은 동물은?';
  $('#matchSubtitle').textContent='동물상 중에서 가장 가까운 이미지를 찾아봤어요.';
  $('#matchIcon').textContent='🐾';
  if(!top3?.length){
    top.innerHTML='<div class="match-empty">동물상 결과를 가져오지 못했어요.</div>';
    list.innerHTML='';
    return;
  }
  const first=top3[0];
  top.innerHTML=topHero(
    `${animalKorean(first.animal)}상`,
    '나와 가장 닮은 동물',
    first.animal_image_url,
    'animal',
    matchScoreText(first)
  );
  list.innerHTML=top3.slice(1).map((m,i)=>rankingItem(m,i,'animal')).join('');
}

function renderPokemonMatches(top3){
  const top=$('#topMatch');
  const list=$('#matchList');
  $('#matchEyebrow').textContent='POKÉMON MATCH';
  $('#matchTitle').textContent='나랑 닮은 포켓몬은?';
  $('#matchSubtitle').textContent='동물상과 얼굴 스타일을 반영해 선택한 포켓몬이에요.';
  $('#matchIcon').textContent='🎮';
  if(!top3?.length){
    top.innerHTML='<div class="match-empty">포켓몬 결과를 가져오지 못했어요.</div>';
    list.innerHTML='';
    return;
  }
  const first=top3[0];
  top.innerHTML=topHero(
    first.pokemon,
    `기반 동물상: ${animalKorean(first.animal)}상`,
    first.pokemon_image_url,
    'pokemon',
    '나와 닮은 포켓몬 TOP 1'
  );
  list.innerHTML=top3.slice(1).map((m,i)=>rankingItem(m,i,'pokemon')).join('');
}

function renderCombinedMatches(top3){
  const top=$('#topMatch');
  const list=$('#matchList');
  $('#matchEyebrow').textContent='LOOKALIKE';
  $('#matchTitle').textContent='나랑 닮은 동물과 포켓몬';
  $('#matchSubtitle').textContent='닮은 동물상과 연결된 포켓몬을 한눈에 만나보세요.';
  $('#matchIcon').textContent='✨';
  if(!top3?.length){
    top.innerHTML='<div class="match-empty">매칭 결과를 가져오지 못했어요.</div>';
    list.innerHTML='';
    return;
  }
  const first=top3[0];
  top.innerHTML=`
    <div class="rank">BEST MATCH</div>
    <div class="match-duo">
      <div class="duo-piece">
        <div class="duo-img animal">
          ${photo(first.animal_image_url,animalKorean(first.animal),'animal','duo-img-real')}
        </div>
        <span class="duo-kicker">가장 닮은 동물</span>
        <strong>${escapeHtml(animalKorean(first.animal))}상</strong>
        <small>${escapeHtml(matchScoreText(first))}</small>
      </div>
      <div class="duo-bridge">✦</div>
      <div class="duo-piece">
        <div class="duo-img pokemon">
          ${photo(first.pokemon_image_url,first.pokemon,'pokemon','duo-img-real')}
        </div>
        <span class="duo-kicker">나랑 닮은 포켓몬</span>
        <strong>${escapeHtml(first.pokemon)}</strong>
        <small>동물상·얼굴 스타일 기반</small>
      </div>
    </div>`;
  list.innerHTML=top3.slice(1).map((m,i)=> `
    <div class="match-item match-compact-result">
      <div>
        <b>#${i+2} ${escapeHtml(animalKorean(m.animal))}상</b>
        <small>🎮 ${escapeHtml(m.pokemon)}</small>
      </div>
      <strong>${escapeHtml(matchScoreText(m))}</strong>
    </div>`).join('');
}

const FORTUNE_TOPIC_ICONS = {
  wealth:'💰', love:'💗', career:'💼', people:'🤝', family:'🏡', late:'🌅'
};

function renderFortuneDetails(f){
  const life=$('#fortuneLife');
  const topicsBox=$('#fortuneTopics');
  life.hidden=true;
  topicsBox.hidden=true;
  const story=(f && f.status==='ready') ? f.life_story : null;
  if(story && story.status==='ready'){
    $('#fortuneLifeHeadline').textContent=story.headline || '내 인생의 관상 이야기';
    $('#fortuneLifeIntro').textContent=story.intro || '';
    $('#fortuneLifeChapters').innerHTML=(story.chapters||[]).map(ch=>`
      <article class="fortune-life-chapter chapter-${escapeHtml(ch.id)}">
        <div class="life-chapter-index">${escapeHtml(ch.index)}</div>
        <div class="life-chapter-copy">
          <span>${escapeHtml(ch.eyebrow)}</span>
          <h5>${escapeHtml(ch.title)}</h5>
          <p>${escapeHtml(ch.text)}</p>
        </div>
      </article>`).join('');
    $('#fortuneLifeNotice').textContent=story.notice || '';
    life.hidden=false;

    const topics=story.topics || [];
    if(topics.length){
      $('#fortuneTopicGrid').innerHTML=topics.map(item=>`
        <details class="fortune-topic-card">
          <summary>
            <span class="fortune-topic-icon">${FORTUNE_TOPIC_ICONS[item.key] || '✦'}</span>
            <span class="fortune-topic-text"><b>${escapeHtml(item.name)}</b>
            <small>${escapeHtml(item.title)}</small></span>
            <span class="fortune-topic-chevron">⌄</span>
          </summary>
          <div class="fortune-topic-detail">
            ${item.body ? `<p>${escapeHtml(item.body)}</p>` : ''}
            ${item.caution ? `<p><b>관상 속 주의점</b> ${escapeHtml(item.caution)}</p>` : ''}
            ${item.advice ? `<p><b>관상 속 조언</b> ${escapeHtml(item.advice)}</p>` : ''}
          </div>
        </details>`).join('');
      topicsBox.hidden=false;
    }
  }
  const eyeCard=$('#fortuneEye');
  const labels=$('#fortuneLabels');
  const labelsTitle=$('#fortuneLabelsTitle');
  const special=$('#fortuneSpecial');
  const traditional=$('#fortuneTraditional');
  const luck=$('#fortuneLuck');
  eyeCard.hidden=true;
  labels.hidden=true;
  labelsTitle.hidden=true;
  special.hidden=true;
  traditional.hidden=true;

  if(!f || f.status!=='ready') return;
  const eye=f.eye_shape||{};
  if(eye.name){
    const similarity=Number(eye.similarity || 0);
    const matchText=Number.isFinite(similarity)?`${Math.round(similarity*100)}점`:'참고용';
    eyeCard.innerHTML=`<span>나의 눈 유형</span><h4>${escapeHtml(eye.distinct?eye.name:'뚜렷한 유형 없음')}</h4>
      <p>${eye.distinct?'가장 가까운 전통 눈 모양: '+escapeHtml(eye.name)+' · 유사 점수 '+escapeHtml(matchText):'특정 눈 유형과의 일치도가 높지 않아요.'}</p>
      <small>${escapeHtml(eye.caveat||'측정된 눈 형태만 비교한 오락용 유사도입니다.')}</small>`;
    eyeCard.hidden=false;
  }
  if(Array.isArray(f.items) && f.items.length){
    labels.innerHTML=f.items.map(x=>`<div class="fortune-label">
      <span>${escapeHtml(x.name)}</span>
      <strong>${escapeHtml(x.label)}</strong>
      <small>${Number(x.level)}/5 단계${x.reliability==='low'?' · 측정 주의':''}</small>
    </div>`).join('');
    labels.hidden=false;
    labelsTitle.hidden=false;
  }
  const rules=f.rules||{};
  if(rules.status==='ok' || rules.status==='no_reference'){
    const htmlParts=[];
    if(rules.ohaeng){
      htmlParts.push(`<div class="fortune-detail-title">오행 얼굴형</div><div class="fortune-detail-body">
        <b>${escapeHtml(rules.ohaeng.name||'')} ${escapeHtml(rules.ohaeng.hanja||'')}</b>
        <p>${escapeHtml(rules.ohaeng.one||'')}</p></div>`);
    }
    if(rules.grade){
      htmlParts.push(`<div class="fortune-detail-title">관상 종합 등급 (오락용)</div>
        <div class="fortune-detail-body"><b>${escapeHtml(rules.grade.grade||'')}</b><p>${escapeHtml(rules.grade.one||'')}</p></div>`);
    }
    if(Array.isArray(rules.zones) && rules.zones.length){
      htmlParts.push('<div class="fortune-detail-title">삼정 · 얼굴 세 구역</div>'+rules.zones.map(z=>
        `<div class="fortune-detail-body"><b>${escapeHtml(z.name||'')}</b> · ${Number(z.value).toFixed(1)}%<p>${escapeHtml(z.text||'')}</p></div>`
      ).join(''));
    }
    special.innerHTML=htmlParts.join('');
    special.hidden=!htmlParts.length;
  }
  const luckList=(f.traditional_luck||[]);
  if(luckList.length){
    const map={low:'부족',mid:'보통',high:'좋음'};
    luck.innerHTML=luckList.map(x=>`<div class="fortune-luck-item"><span>${escapeHtml(x.name||'')}</span>
      <strong>${escapeHtml(map[x.state]||x.state||'')}</strong></div>`).join('')+
      '<p class="fortune-luck-warning">관상 속설을 흉내 낸 엔터테인먼트 콘텐츠입니다. 건강 상태나 실제 미래를 예측하지 않습니다.</p>';
    traditional.hidden=false;
  }
}

function applyResultMode(mode){
  if(!state.result || !MODE_INFO[mode]) return;
  state.mode = mode;
  const r = state.result;
  const matchCard = $('#matchCard');
  const metricsCard = $('#metricsCard');
  const fortuneCard = $('#fortuneCard');
  const resultBottom = $('#resultBottom');
  resultZone.dataset.activeMode=mode;

  matchCard.hidden = false;
  metricsCard.hidden = false;
  fortuneCard.hidden = false;
  resultBottom.hidden = false;

  $$('[data-result-mode]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.resultMode === mode);
  });

  if(mode === 'fortune'){
    $('#resultIcon').textContent = '🔮';
    $('#resultEyebrow').textContent = 'FACE READING';
    $('#resultTitle').textContent = '내 관상은 어떨까요?';
    $('#resultSubtitle').textContent = '내 얼굴의 특징을 바탕으로 전통 관상에서는 어떻게 풀이하는지 살펴봐요.';
    matchCard.hidden = true;
    metricsCard.hidden = true;
  }else if(mode === 'animal'){
    $('#resultIcon').textContent = '🐾';
    $('#resultEyebrow').textContent = 'ANIMAL MATCH';
    $('#resultTitle').textContent = '나랑 가장 닮은 동물은?';
    $('#resultSubtitle').textContent = '내 얼굴 특징과 가장 가까운 동물을 찾아봤어요.';
    resultBottom.hidden = true;
    renderAnimalMatches(r.top3 || []);
  }else if(mode === 'pokemon'){
    $('#resultIcon').textContent = '🎮';
    $('#resultEyebrow').textContent = 'POKÉMON MATCH';
    $('#resultTitle').textContent = '나랑 닮은 포켓몬은?';
    $('#resultSubtitle').textContent = '얼굴의 분위기와 동물 분석 결과를 바탕으로 찾아봤어요.';
    resultBottom.hidden = true;
    renderPokemonMatches(r.top3 || []);
  }else{
    $('#resultIcon').textContent = '✨';
    $('#resultEyebrow').textContent = 'ALL IN ONE';
    $('#resultTitle').textContent = '내 얼굴 분석이 완료됐어요!';
    $('#resultSubtitle').textContent = '내 얼굴 특징부터 관상, 동물, 포켓몬까지 한 번에 확인해보세요.';
    renderCombinedMatches(r.top3 || []);
    renderMetrics(r.metrics || []);
  }

  $('#selectedModePill').textContent = `${MODE_INFO[mode].icon} ${MODE_INFO[mode].label}`;
}

$$('[data-result-mode]').forEach(btn => {
  btn.addEventListener('click', () => {
    applyResultMode(btn.dataset.resultMode);
    $('#resultZone').scrollIntoView({behavior:'smooth', block:'start'});
  });
});

async function loadResult(){
  state.loadingResult = true;
  try{
    const r = await jsonFetch(resultApiUrl());
    if(!r.ready) return;

    state.result = r;
    $('#resultImage').src = `${r.image_url}?t=${Date.now()}`;
    $('#captureId').textContent = r.capture_id;
    renderFortuneDetails(r.fortune);

    $('#fortuneLines').innerHTML =
      (r.fortune_lines || [])
        .map(x => `<div class="fortune-line">${escapeHtml(x)}</div>`)
        .join('') ||
      '<div class="fortune-line">얼굴 이야기를 준비하고 있어요.</div>';

    scanFlow.hidden = true;
    resultZone.hidden = false;
    applyResultMode(state.mode || 'all');

    setTimeout(() => resultZone.scrollIntoView({behavior:'smooth', block:'start'}), 120);
    showToast('분석이 완료됐어요 ✨');
  }catch(err){
    showToast(`결과 표시 실패: ${err.message}`, true);
  }finally{
    state.loadingResult = false;
  }
}

if(window.location.pathname === '/preview-result'){
  // Use already-saved data, no camera scan needed.
  jsonFetch(resultApiUrl()).then(r=>{
    if(!r.ready) return showToast('저장된 분석 결과가 없습니다.',true);
    state.result=r;
    state.lastCapture=r.capture_id;
    state.mode='all';
    homeSection.hidden=true;
    scanFlow.hidden=true;
    resultZone.hidden=false;
    homeBtn.hidden=false;
    $('#resultImage').src=`${r.image_url}?t=${Date.now()}`;
    $('#captureId').textContent=r.capture_id;
    renderFortuneDetails(r.fortune);
    $('#fortuneLines').innerHTML=(r.fortune_lines||[])
      .map(x=>`<div class="fortune-line">${escapeHtml(x)}</div>`).join('');
    applyResultMode('all');
    setTimeout(()=>resultZone.scrollIntoView({behavior:'smooth',block:'start'}),150);
  }).catch(err=>showToast('저장된 결과를 불러오지 못했어요: '+err.message,true));
}
pollStatus();
setInterval(pollStatus,700);
