import { parseInstruction } from './language-parser.mjs';
const $ = id => document.getElementById(id);
const val = id => $(id).value.trim();
let reviewedCommand = null;
function configuredBowls() {
  return [{ id: 'bowl_1', food: $('bowl-1-food').value }, { id: 'bowl_2', food: $('bowl-2-food').value }];
}
function clearReview() {
  reviewedCommand = null; $('copy').disabled = true; $('copy-status').textContent = '';
  $('intent').textContent = 'Waiting for an instruction.'; $('raw').textContent = 'No command checked yet.';
  $('mapping').textContent = 'Review only · nothing has been sent.';
  $('preview-title').textContent = 'Command preview';
  $('copy').textContent = 'Copy command JSON';
}
function interpret(text) {
  const result = parseInstruction(text, configuredBowls());
  if (!result.ok) throw new Error(result.explanation);
  return result;
}
function presentInterpretation(result) {
  clearReview(); reviewedCommand = result.request || result.command;
  $('intent').textContent = JSON.stringify(result.intent, null, 2);
  $('raw').textContent = reviewedCommand ? JSON.stringify(reviewedCommand, null, 2) : 'No robot command mapping yet.';
  $('mapping').textContent = result.command ? 'Existing dashboard command · preview only, not sent.' : result.mapping;
  if (result.request) {
    $('preview-title').textContent = 'Feeding request preview';
    $('copy').textContent = 'Copy request JSON';
    $('mapping').textContent = result.mapping;
  }
  $('copy').disabled = !reviewedCommand;
  say('ok', result.explanation + ' Preview only. Nothing was sent to the robot.');
}
function nlParse() {
  invalidateVoice(); micUp(); clearReview();
  try { presentInterpretation(interpret(val('nltext'))); return true; }
  catch (e) { say('bad', e.message); return false; }
}
function say(kind, text) { $('nlexplain').dataset.kind = kind; $('nlexplain').textContent = text; }
function tryEx(button) { $('nltext').value = button.textContent; nlParse(); }
function toggleTheme() { const root = document.documentElement; root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light'; }
async function copyCommand() {
  const command = reviewedCommand; if (!command) return;
  try { await navigator.clipboard.writeText(JSON.stringify(command, null, 2));
    if (reviewedCommand === command) $('copy-status').textContent = ' Copied. Not sent.';
  } catch { if (reviewedCommand === command) $('copy-status').textContent = ' Select the JSON to copy it manually.'; }
}
// Voice controls adapted from the existing Ultraviolet hackathon dashboard.
// Speech recognisers return several guesses; try them in order
// and keep the first one the parser understands (helps with accents and misheard words).
const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, guesses = [], micMode = null, micStopping = false;
const synthesis = window.speechSynthesis && typeof window.SpeechSynthesisUtterance === 'function' ? window.speechSynthesis : null;
let voiceGeneration = 0, replyGeneration = 0, currentReply = null;
let replyVoices = [], savedReplyVoice = '';
try { savedReplyVoice = localStorage.getItem('feeding-uv-reply-voice') || ''; } catch (e) {}
// Exclude character/effect voices from normal speech choices.
const effectVoice = /^(Albert|Bad News|Bahh|Bells|Boing|Bubbles|Cellos|Fred|Good News|Grandma|Grandpa|Jester|Junior|Organ|Ralph|Superstar|Trinoids|Whisper|Wobble|Zarvox)(\b|$)/i;
function voiceScore(voice) {
  const quality = /natural|enhanced|premium|neural/i.test(voice.name) ? 1000
    : /google us english/i.test(voice.name) ? 900 : /google.*english/i.test(voice.name) ? 800
    : /\b(Samantha|Daniel|Karen|Moira|Tessa|Aaron|Arthur)\b/i.test(voice.name) ? 600 : voice.default ? 400 : 100;
  return quality + (/^en-US$/i.test(voice.lang) ? 20 : 0) + (voice.default ? 10 : 0);
}
function selectedReplyVoice() {
  return replyVoices.find(v => v.voiceURI === $('reply-voice').value) || replyVoices[0];
}
function updateVoiceControls() {
  $('preview-voice').disabled = !synthesis || !!rec || !$('spoken-replies').checked;
  $('reply-voice').disabled = !synthesis || !!rec;
  const voice = selectedReplyVoice();
  $('voice-source').textContent = voice ? (voice.localService ? 'On-device voice' : 'Online voice · needs internet') : 'Browser default voice';
}
function refreshReplyVoices() {
  try { replyVoices = synthesis ? synthesis.getVoices().filter(v => /^en(?:-|$)/i.test(v.lang) && !effectVoice.test(v.name)) : []; }
  catch (e) { replyVoices = []; }
  replyVoices.sort((a, b) => voiceScore(b) - voiceScore(a) || a.name.localeCompare(b.name));
  const select = $('reply-voice'), automatic = document.createElement('option');
  automatic.value = ''; automatic.textContent = replyVoices[0] ? 'Automatic — ' + replyVoices[0].name : 'Automatic';
  select.replaceChildren(automatic);
  replyVoices.forEach(v => { const option = document.createElement('option'); option.value = v.voiceURI;
    option.textContent = v.name + ' · ' + v.lang + (v.localService ? '' : ' · online'); select.appendChild(option); });
  select.value = replyVoices.some(v => v.voiceURI === savedReplyVoice) ? savedReplyVoice : '';
  updateVoiceControls();
}
function changeReplyVoice() {
  savedReplyVoice = $('reply-voice').value; stopSpokenReply();
  try { localStorage.setItem('feeding-uv-reply-voice', savedReplyVoice); } catch (e) {}
}
function previewReplyVoice() {
  if (rec || !$('spoken-replies').checked) return;
  invalidateVoice();
  speakReply("Hey there. Tell me what you'd like to eat, and I'll read it back. This is a local preview.", voiceGeneration, replyGeneration);
}
function stopSpokenReply() {
  replyGeneration++; currentReply = null;
  try { synthesis?.cancel(); } catch (e) { /* text feedback remains available */ }
  $('voice-reply-status').textContent = !synthesis ? 'Spoken replies unavailable in this browser.'
    : $('spoken-replies').checked ? 'Voice replies on.' : 'Voice replies muted.';
  updateVoiceControls();
}
function invalidateVoice() {
  voiceGeneration++; stopSpokenReply();
  if ($('nlexplain').dataset.kind === 'listen') say('idle', 'Voice input stopped. Enter a command or start listening again.');
}
function toggleSpokenReplies() {
  stopSpokenReply();
  try { localStorage.setItem('feeding-uv-spoken-replies', String($('spoken-replies').checked)); } catch (e) {}
}
function currentVoice(token) {
  return token === voiceGeneration && !rec && !document.hidden && $('panel-Run').classList.contains('on');
}
function speakReply(message, token, replyToken) {
  if (!synthesis || !$('spoken-replies').checked || !currentVoice(token) || replyToken !== replyGeneration) return;
  try {
    const utterance = new window.SpeechSynthesisUtterance(message); currentReply = utterance;
    utterance.lang = document.documentElement.lang || 'en';
    // Keep the voice's natural pitch and a lightly quicker conversational pace.
    utterance.rate = 1.08; utterance.pitch = 1;
    refreshReplyVoices(); const voice = selectedReplyVoice();
    if (voice) { utterance.voice = voice; utterance.lang = voice.lang; }
    utterance.onend = () => { if (currentReply === utterance) { currentReply = null; $('voice-reply-status').textContent = 'Spoken reply finished.'; } };
    utterance.onerror = () => { if (currentReply === utterance) { currentReply = null; $('voice-reply-status').textContent = 'Audio unavailable. Read the response below.'; } };
    $('voice-reply-status').textContent = 'Replying…'; synthesis.speak(utterance);
  } catch (e) { currentReply = null; $('voice-reply-status').textContent = 'Audio unavailable. Read the response below.'; }
}
function spokenInterpretation(result) { return 'Got it. ' + result.explanation + ' Preview only. Nothing was sent to the robot.'; }
try { $('spoken-replies').checked = localStorage.getItem('feeding-uv-spoken-replies') !== 'false'; } catch (e) {}
if (!synthesis) { $('spoken-replies').checked = false; $('spoken-replies').disabled = true; }
if (synthesis) synthesis.addEventListener('voiceschanged', refreshReplyVoices);
refreshReplyVoices();
stopSpokenReply();
$('nltext').addEventListener('input', () => { invalidateVoice(); micUp(); clearReview(); say('idle', 'Command changed. Press Check to review it.'); });
function micIdle(session) {
  if (rec !== session) return false;
  rec = null; micMode = null; micStopping = false;
  $('mic').classList.remove('on'); $('mic').textContent = '🎤 Hold to talk';
  $('mic').setAttribute('aria-pressed', 'false');
  updateVoiceControls();
  return true;
}
function micDown(e) {
  if (e.button !== 0 || e.isPrimary === false) return;
  e.preventDefault();
  if (rec) { micUp(); return; }
  micStart('pointer');
}
// Native button activation supplies keyboard and assistive-technology clicks.
// Ignore the pointer click after release so a hold never restarts recognition.
function micClick(e) { if (e.detail === 0) { if (rec) micUp(); else micStart('toggle'); } }
function micStart(mode) {
  if (rec) return;
  invalidateVoice(); clearReview(); const token = voiceGeneration, replyToken = replyGeneration;
  if (!Speech) { say('bad', 'Voice input needs Chrome or Edge; this browser has no speech recognition. You can still type.'); return; }
  let session;
  try { session = new Speech(); } catch (e) { say('bad', 'Voice input could not start: ' + e.message); return; }
  guesses = []; rec = session; micMode = mode; micStopping = false;
  updateVoiceControls();
  session.lang = navigator.language || 'en-US'; session.interimResults = true; session.continuous = true; session.maxAlternatives = 5;
  session.onresult = ev => {
    if (rec !== session || token !== voiceGeneration) return;
    const finals = [...ev.results].filter(r => r.isFinal);
    guesses = finals.length === 1 ? [...finals[0]].map(a => a.transcript.trim())
      : [finals.map(r => r[0].transcript.trim()).join(' ')];
    $('nltext').value = [...ev.results].map(r => r[0].transcript).join(' ').trim();
  };
  session.onerror = ev => { if (!micIdle(session) || !currentVoice(token)) return; say('bad', 'Voice error: ' + ev.error
    + (ev.error === 'not-allowed' ? ' (allow the microphone for this page)' : ev.error === 'network' ? ' (this browser cannot reach its speech service; try Chrome)' : '')); };
  session.onend = () => { if (micIdle(session) && currentVoice(token)) voiceParse(token, replyToken); };
  $('mic').classList.add('on'); $('mic').textContent = mode === 'toggle' ? '🎙 Stop listening' : '🎙 Listening…';
  $('mic').setAttribute('aria-pressed', 'true');
  say('listen', mode === 'toggle' ? 'Listening… press Enter or Space again to finish.' : 'Listening… let go of the button when you finish.');
  try { session.start(); } catch (e) { if (micIdle(session)) say('bad', 'Voice input could not start: ' + e.message); }
}
function micUp(mode) {
  if (!rec || micStopping || (mode && micMode !== mode)) return;
  const session = rec; micStopping = true;
  try { session.stop(); } catch (e) { if (micIdle(session)) say('bad', 'Voice input could not stop: ' + e.message); }
}
document.addEventListener('visibilitychange', () => { if (document.hidden) { invalidateVoice(); micUp(); } });
window.addEventListener('pagehide', () => { invalidateVoice(); micUp(); });
async function voiceParse(token, replyToken) {
  const options = guesses.filter(Boolean);
  if (!currentVoice(token)) return;
  if (!options.length) {
    const message = "I missed that one! Could you say it again, or type it in?";
    say('bad', message); speakReply(message, token, replyToken); return;
  }
  let chosen, result, error;
  for (const text of options) {
    try { result = interpret(text); chosen = text; }
    catch (e) { error = e; }
    if (!currentVoice(token)) return;
    if (result) break;
  }
  if (!result) {
    say('bad', "Sorry, I didn't understand that. " + (error?.message || 'Please try again.'));
    speakReply("Hmm, I don't know that command yet. Try asking for a bite, a scoop, or to go home.", token, replyToken); return;
  }
  $('nltext').value = chosen; presentInterpretation(result);
  say($('nlexplain').dataset.kind, `🎤 heard "${chosen}"` + (options.length > 1 && chosen !== options[0] ? ` (best of ${options.length} guesses)` : '') + ' — ' + $('nlexplain').textContent);
  speakReply(spokenInterpretation(result), token, replyToken);
}

Object.assign(window, { nlParse, tryEx, toggleTheme, copyCommand, micDown, micUp, micClick, toggleSpokenReplies, changeReplyVoice, previewReplyVoice });

function showView(view) {
  const manual = view === 'manual';
  invalidateVoice(); micUp();
  $('panel-Run').hidden = manual; $('panel-Run').classList.toggle('on', !manual);
  $('manual-panel').hidden = !manual;
  for (const [id, selected] of [['language-tab', !manual], ['manual-tab', manual]]) {
    $(id).classList.toggle('on', selected); $(id).setAttribute('aria-pressed', String(selected));
  }
  if (manual && !$('manual-dashboard').hasAttribute('src')) $('manual-dashboard').src = 'face-servo.html?preview=1';
}
window.showView = showView;
for (const id of ['bowl-1-food', 'bowl-2-food']) $(id).addEventListener('change', () => {
  invalidateVoice(); micUp(); clearReview();
  say('idle', 'Bowl labels changed. Check your instruction again to select the correct bowl.');
});
