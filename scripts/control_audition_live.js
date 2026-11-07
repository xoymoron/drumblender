/* Fixed audition cards stay intact; new draws never filter inactive features. */
(async () => {
  const response = await fetch('/api/state', {cache: 'no-store'});
  if (!response.ok) return;
  const state = await response.json();
  const page = location.pathname.split('/').pop() || 'index.html';
  const model = page.startsWith('baseline') ? 'comparison' : 'main';
  const scope = page === 'modal-only.html' ? 'modal-only' :
    page.includes('modal') ? 'modal' : page.includes('noise') ? 'noise' : 'mix';
  const storageKey = `control-choice:${state.sae_sha256}:${model}`;
  let busy = false;
  const note = document.createElement('p');
  note.textContent = `Live validation audition: ${state.validation_count} samples. Fixed examples below are selected for activation; random draws are unfiltered. No retraining. Negative results stay visible. Play 0, then ±4 or ±16. New draws may take several seconds.`;
  document.querySelector('.deck').after(note);
  const saved = JSON.parse(localStorage.getItem(storageKey) || '[]');
  document.querySelectorAll('input[type=checkbox]').forEach(box => {
    box.checked = saved.includes(Number(box.value));
  });
  const showChoices = () => {
    const selected = [...document.querySelectorAll('input:checked')].map(x => Number(x.value));
    localStorage.setItem(storageKey, JSON.stringify(selected));
    document.querySelector('output').textContent = 'Selected atoms: ' + (selected.join(', ') || '—');
  };
  showChoices();
  document.addEventListener('change', showChoices);
  function show(box, row) {
    box.replaceChildren();
    const name = document.createElement('p');
    name.textContent = `${row.name} · pack ${row.pack} · ${row.active ? 'feature active' : 'FEATURE INACTIVE: identical results are expected'} · ${row.seconds.toFixed(2)} s`;
    box.append(name);
    for (const [label, path] of Object.entries(row.audio[scope])) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.audio = path;
      button.textContent = label;
      box.append(button, ' ');
    }
  }
  for (const section of document.querySelectorAll('section')) {
    const atom = Number(section.querySelector('input').value);
    const panel = document.createElement('div');
    panel.className = 'live-panel';
    panel.style.cssText = 'border-top:2px solid #c7e76d;margin-top:14px;padding-top:12px';
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = '새 무작위 sample';
    const status = document.createElement('p');
    status.textContent = 'Validation 전체에서 균등 추출 · 고정 3개 제외 · 중복 없이 순회';
    const audio = document.createElement('div');
    panel.append(button, status, audio);
    section.append(panel);
    const previous = state.draws.find(row => row.atom === atom && row.model === model);
    if (previous) show(audio, previous);
    button.addEventListener('click', async () => {
      if (busy) { status.textContent = '다른 렌더가 끝난 뒤 다시 눌러줘.'; return; }
      busy = true;
      document.querySelectorAll('.live-panel button:not([data-audio])').forEach(b => b.disabled = true);
      status.textContent = '새 sample 추론 중… 완료되면 아래에 재생 버튼이 표시돼.';
      try {
        const result = await fetch('/api/render', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({atom, model})
        });
        const row = await result.json();
        if (!result.ok) throw new Error(row.error || `HTTP ${result.status}`);
        show(audio, row);
        status.textContent = '무작위 추출 완료. 소리가 안 바뀌는 경우도 이 feature의 결과야.';
      } catch (error) { status.textContent = `렌더 실패: ${error.message}`; }
      finally {
        busy = false;
        document.querySelectorAll('.live-panel button:not([data-audio])').forEach(b => b.disabled = false);
      }
    });
  }
})().catch(error => console.error('Live audition unavailable:', error));
