import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseInstruction as parse } from '../static/language-parser.mjs';
for (const [text, action] of [
  ['feed me a bite','feed'], ['Could you please feed me another bite?','feed'],
  ['scoop some food','scoop'], ['go home','home'], ['find my face','find_face'],
  ['center on my face','center'], ['level the spoon','level_spoon'], ['retreat','retreat'],
]) test(text, () => assert.deepEqual(parse(text).command, {action}));
test('safety words win in mixed instructions', () => {
  for (const text of ['feed me a bite, stop', 'no, feed me a bite', 'wait then scoop', "I'm done, go home", 'too close', 'ouch'])
    assert.deepEqual(parse(text).command, {action:'stop'});
});
test('food selection is preserved, never dropped into a generic feed command', () => {
  const r=parse('feed me the strawberries'); assert.equal(r.intent.item,'strawberry'); assert.equal(r.command,null);
});
test('preferences, drink, query and confirmation have no invented robot mapping', () => {
  for (const text of ['slower','smaller bite','a sip of water',"what's on the plate?",'yes']) {
    assert.equal(parse(text).ok,true); assert.equal(parse(text).command,null);
  }
});
test('whole-command grammar rejects extra meaning and unknown foods', () => {
  for (const text of ['feed me a bite and go home', 'scoop twice', 'feed me peanuts', "don't feed me",'go home faster','go home unless ready','move 20 cm','<script>alert(1)</script>','x'.repeat(301),''])
    assert.equal(parse(text).ok,false,text);
});
test('safety words use boundaries and handle curly apostrophes', () => {
  assert.equal(parse('yogurt').intent.item,'yogurt'); assert.equal(parse('I’m done').intent.action,'done');
});
test('offline UI has no robot transport or background requests', () => {
  const app=readFileSync(new URL('../static/language-app.js',import.meta.url),'utf8');
  const page=readFileSync(new URL('../static/language.html',import.meta.url),'utf8');
  assert.doesNotMatch(app,/\bfetch\s*\(|WebSocket|XMLHttpRequest|\/api\/|viam-connection|doCommand/);
  assert.match(page,/id="send" disabled/);
  assert.doesNotMatch(page,/stream\.mjpg|\/api\/|<iframe[^>]*src=/);
  assert.match(app,/clearReview\(\); say\('idle', 'Command changed/);
});
test('original camera panels and movement controls are retained behind offline preview', () => {
  const original=readFileSync(new URL('../static/face-servo.html',import.meta.url),'utf8');
  for (const action of ['teleop_start','teleop_stop','teach_look','teach_home','teach_close','hand_guide_on','hand_guide_off','home','search','sweep','find_face','plan_scoop','scoop','home_scoop','level_spoon','feed','auto_center','center','extend','retreat'])
    assert(original.includes(`data-a="${action}"`),action);
  for(const id of ['cam','depthcam','stop','consent']) assert(original.includes(`id="${id}"`));
  assert.match(original,/if \(preview\) return;/);
  assert.match(original,/else poll\(\);/);
  assert.doesNotMatch(original,/<iframe[^>]*src=/);
});
test('cereal request resolves to its labeled bowl with an explicit request preview', () => {
  for (const text of ['I want some cereal', 'Can I have cereal please?', "I'd like a spoonful of cereal", 'feed me cereal']) {
    const r=parse(text);
    assert.equal(r.ok,true,text); assert.equal(r.intent.item,'cereal');
    assert.equal(r.intent.source_bowl,'bowl_1');
    assert.equal(r.command,null);
    assert.equal(r.request.kind,'feeding_request_preview');
    assert.deepEqual(r.request.steps,['scoop_from_selected_bowl','confirm_readiness','approach_mouth','return_home']);
  }
});
test('M&Ms written and spoken forms select the second bowl', () => {
  for(const text of ['I want some M&Ms', "give me M&M's", 'can I have m and ms', 'some em and ems', 'MMS']) {
    const r=parse(text); assert.equal(r.ok,true,text); assert.equal(r.request.source_bowl,'bowl_2'); assert.equal(r.request.food,'m&ms');
  }
});
test('bowl selection follows configuration rather than assuming a physical side', () => {
  const bowls=[{id:'bowl_1',food:'m&ms'},{id:'bowl_2',food:'cereal'}];
  assert.equal(parse('I want cereal',bowls).request.source_bowl,'bowl_2');
  assert.equal(parse('M&Ms',bowls).request.source_bowl,'bowl_1');
});
test('missing and ambiguous bowls need clarification', () => {
  for(const bowls of [
    [{id:'bowl_1',food:''},{id:'bowl_2',food:'m&ms'}],
    [{id:'bowl_1',food:'cereal'},{id:'bowl_2',food:'cereal'}],
    [{id:'unknown',food:'cereal'}], null,
  ]) { const r=parse('I want cereal',bowls); assert.equal(r.ok,false); assert.equal(r.command,null); }
});
test('bowl requests preserve the full request and never ignore qualifiers', () => {
  for(const text of ['cereal and M&Ms','I want cereal twice','I want cereal from the left bowl','I want three bites of cereal'])
    assert.equal(parse(text).ok,false,text);
  for(const text of ['I want cereal, wait','no M&Ms','stop, I want cereal']) {
    const r=parse(text); assert.deepEqual(r.command,{action:'stop'}); assert.equal(r.request,undefined);
  }
});
test('changing bowl labels invalidates the previous review and voice generation', () => {
  const app=readFileSync(new URL('../static/language-app.js',import.meta.url),'utf8');
  assert.match(app,/for \(const id of \['bowl-1-food', 'bowl-2-food'\]\).*addEventListener\('change', \(\) => \{\s*invalidateVoice\(\); micUp\(\); clearReview\(\);/);
});
