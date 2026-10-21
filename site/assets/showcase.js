/* Browser-local presentation only; no inference or localhost requests. */
'use strict';
for (const button of document.querySelectorAll('[data-copy]')) {
  button.addEventListener('click', async () => {
    const status = button.closest('.code-panel').querySelector('.copy-status');
    try {
      await navigator.clipboard.writeText(document.getElementById(button.dataset.copy).textContent);
      status.textContent = document.documentElement.lang === 'zh-CN' ? '已复制命令' : 'Commands copied';
    } catch {
      status.textContent = document.documentElement.lang === 'zh-CN' ? '请选中下方命令复制' : 'Select the commands below to copy';
    }
  });
}
const tabs = [...document.querySelectorAll('[data-stage]')];
function selectStage(tab) {
  for (const item of tabs) {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    document.getElementById(item.getAttribute('aria-controls')).hidden = !selected;
  }
}
for (const [index, tab] of tabs.entries()) {
  tab.addEventListener('click', () => selectStage(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = tabs[(index + 1) % tabs.length];
    if (event.key === 'ArrowLeft') next = tabs[(index + tabs.length - 1) % tabs.length];
    if (event.key === 'Home') next = tabs[0];
    if (event.key === 'End') next = tabs[tabs.length - 1];
    if (next) { event.preventDefault(); selectStage(next); next.focus(); }
  });
}
const play = document.getElementById('replay-play');
if (play) {
  const output = document.getElementById('replay-output');
  const status = document.getElementById('replay-status');
  const speed = document.getElementById('replay-speed');
  let replay, timer, playing = false, position = 0, elapsed = 0, started = 0, currentSpeed = 4;
  function stop() {
    if (playing) elapsed += (performance.now() - started) * currentSpeed;
    playing = false;
    clearTimeout(timer);
    play.textContent = 'Resume recording';
    speed.disabled = false;
  }
  function tick() {
    const time = elapsed + (performance.now() - started) * currentSpeed;
    while (position < replay.events.length && replay.events[position].elapsed_ms <= time) {
      output.textContent += replay.events[position++].text;
    }
    status.textContent = `Recorded ${(time / 1000).toFixed(1)} s · ${currentSpeed}× playback`;
    if (position === replay.events.length) {
      stop();
      play.textContent = 'Replay recording';
      status.textContent = 'Recording complete · source response preserved';
    } else timer = setTimeout(tick, 30);
  }
  play.addEventListener('click', async () => {
    if (playing) { stop(); status.textContent = 'Recording paused'; return; }
    play.disabled = true;
    try {
      replay ??= await fetch('data/recorded-response.json').then(response => {
        if (!response.ok) throw new Error('Recording unavailable');
        return response.json();
      });
      if (position === 0 || position === replay.events.length) {
        output.textContent = ''; position = 0; elapsed = 0;
      }
      currentSpeed = Number(speed.value);
      started = performance.now(); playing = true; speed.disabled = true;
      play.textContent = 'Pause recording'; tick();
    } catch {
      status.textContent = 'Recording could not load. Open the source JSON below.';
    } finally { play.disabled = false; }
  });
  document.getElementById('replay-reset').addEventListener('click', () => {
    stop(); position = 0; elapsed = 0;
    output.textContent = 'Press play to replay the captured response.';
    status.textContent = 'Ready · recorded baseline response';
    play.textContent = 'Play recording';
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && playing) { stop(); status.textContent = 'Recording paused'; }
  });
}
