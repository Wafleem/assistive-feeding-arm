// Local, bounded command translation for the feeding dashboard. No transport.
// Reuses Ultraviolet's check -> review flow, with the existing SO-101 action names.
const result = (action, explanation, command = null, fields = {}) => ({
  ok: true, intent: { action, ...fields }, command, explanation,
  mapping: command ? '' : 'This intent needs a feeding-controller integration. No command is generated.',
});
const unknown = explanation => ({ ok: false, explanation, intent: null, command: null });
const motion = (action, explanation) => result(action, explanation, { action });
// Logical UI labels only; these do not imply taught robot positions.
export const DEFAULT_BOWLS = Object.freeze([
  Object.freeze({ id: 'bowl_1', food: 'cereal' }),
  Object.freeze({ id: 'bowl_2', food: 'm&ms' }),
]);
const bowlFoods = new Map([
  ['cereal', 'cereal'], ['cereals', 'cereal'],
  ['m&ms', 'm&ms'], ["m&m's", 'm&ms'], ['m & ms', 'm&ms'],
  ['m and ms', 'm&ms'], ["m and m's", 'm&ms'], ['m and m s', 'm&ms'],
  ['m and ems', 'm&ms'], ['em and ems', 'm&ms'], ['mms', 'm&ms'],
]);
function bowlRequest(food, bowls) {
  if (!Array.isArray(bowls) || bowls.length !== 2 ||
      bowls.some((b, i) => !b || b.id !== `bowl_${i + 1}` || !['cereal', 'm&ms', ''].includes(b.food)))
    return unknown('Set the foods for Bowl 1 and Bowl 2 before checking this request.');
  const matches = bowls.filter(b => b.food === food);
  const label = food === 'm&ms' ? 'M&Ms' : 'cereal';
  if (!matches.length) return unknown(`No bowl is labeled ${label}. Update the bowl labels first.`);
  if (matches.length > 1) return unknown(`Both bowls are labeled ${label}. Give each food one bowl before continuing.`);
  const bowl = matches[0], number = bowl.id === 'bowl_1' ? 1 : 2;
  return {
    ...result('feed', `One bite of ${label} from Bowl ${number}: scoop, confirm the spoon is loaded and you are ready, bring the spoon toward the mouth, then return home.`,
      null,
      { item: food, source_bowl: bowl.id, count: 1 }),
    request: {
      kind: 'feeding_request_preview', food, source_bowl: bowl.id, bites: 1,
      steps: ['scoop_from_selected_bowl', 'confirm_readiness', 'approach_mouth', 'return_home'],
    },
    mapping: 'Request preview only. The controller on main is unchanged; this is not an executable robot command.',
  };
}
const foods = new Map([
  ['grape', 'grape'], ['grapes', 'grape'], ['strawberry', 'strawberry'], ['strawberries', 'strawberry'],
  ['berry', 'strawberry'], ['berries', 'strawberry'], ['melon', 'melon cube'], ['melon cube', 'melon cube'],
  ['melon cubes', 'melon cube'], ['watermelon', 'melon cube'], ['cheese', 'cheese cube'], ['cheese cube', 'cheese cube'],
  ['cracker', 'cracker'], ['crackers', 'cracker'], ['yogurt', 'yogurt'], ['yoghurt', 'yogurt'],
  ['pudding', 'yogurt'], ['applesauce', 'yogurt'],
]);
export function parseInstruction(input, bowls = DEFAULT_BOWLS) {
  if (typeof input !== 'string' || !input.trim()) return unknown('Type or say one feeding instruction.');
  if (input.length > 300) return unknown('Please use one short instruction, up to 300 characters.');
  let text = input.toLowerCase().replace(/[’‘]/g, "'").replace(/\s+/g, ' ').trim();
  // Whole words: “wait” wins in a mixed utterance; “yogurt” never contains a stop token.
  if (/\b(stop|halt|freeze|cancel|abort|ow|ouch)\b|\b(too close|back off)\b/.test(text))
    return result('stop', 'Stop requested.', { action: 'stop' });
  if (/\b(wait|pause)\b|\bhold on\b/.test(text))
    return result('pause', 'Pause requested. The dashboard uses STOP to halt motion.', { action: 'stop' });
  if (/\b(i'?m done|i am done|i'?m full|i am full|finish the meal)\b/.test(text))
    return result('done', 'End the meal and stop.', { action: 'stop' });
  if (/\bno\b/.test(text)) return result('confirm', 'No: cancel the pending action and stop.', { action: 'stop' }, { answer: false });
  if (/\b(not|never|dont|don't|cannot|can't|without|except|unless)\b/.test(text))
    return unknown('That instruction includes a condition or negation. Please give one clear command.');
  text = text.replace(/[.!?]+$/, '').replace(/^(?:(?:can|could|would) you (?:please )?|please )/, '')
    .replace(/(?:,? please|,? thanks|,? thank you)$/, '').trim();
  const requestedFood = text.replace(/^(?:i want|i would like|i'd like|can i have|could i have|feed me|give me|get me) /, '')
    .replace(/^(?:(?:a|one) (?:bite|spoonful) of |some |the )/, '');
  if (bowlFoods.has(requestedFood)) return bowlRequest(bowlFoods.get(requestedFood), bowls);
  if (/^(feed me(?: (?:a|one|the|another|next) bite)?|(?:next|another|one more) bite|a bite)$/.test(text))
    return motion('feed', 'Prepare one feeding cycle. Readiness confirmation is required before any live bite.');
  if (/^(scoop(?: (?:some )?food| (?:a|one) bite)?|scoop and come back)$/.test(text))
    return motion('scoop', 'Scoop food and come back.');
  if (/^(?:home|(?:go|return|go back) (?:to )?home|start centered)$/.test(text))
    return motion('home', 'Return to the taught start pose.');
  if (/^(?:(?:sweep|check|scan|identify|look (?:at|in|into)) (?:the |both )?bowls?|what(?:'s| is) in (?:the |each |both )?bowls?)$/.test(text))
    return motion('sweep_bowls', 'Sweep over the bowls: the camera looks into each and CLIP on the Orin says what is in it.');
  if (/^(find (?:my|the) face|look for (?:my|the) face)$/.test(text))
    return motion('find_face', 'Find the face near the taught look pose.');
  if (/^(center(?: on (?:my|the) face)?|centre(?: on (?:my|the) face)?)$/.test(text))
    return motion('center', 'Center on the face already in view.');
  if (/^(level (?:the )?spoon|keep (?:the )?spoon level)$/.test(text))
    return motion('level_spoon', 'Level the spoon using the saved calibration.');
  if (/^(retreat|move back|back away)$/.test(text)) return motion('retreat', 'Retreat to the taught look pose.');
  if (/^(slower|slow down|go slower|faster|speed up|go faster)$/.test(text)) {
    const pace = /slower|slow down/.test(text) ? 'slower' : 'faster';
    return result('adjust', `Requested a ${pace} pace.`, null, { adjust: { pace } });
  }
  if (/^(?:a )?(smaller|bigger|larger)(?: bites?| bite size)$/.test(text)) {
    const bite_size = text.includes('smaller') ? 'smaller' : 'bigger';
    return result('adjust', `Requested ${bite_size} bites.`, null, { adjust: { bite_size } });
  }
  if (/^(a sip of water|(?:a )?(?:sip|drink)(?: of water)?|water|give me (?:a sip of )?water)$/.test(text))
    return result('drink', 'Requested a drink of water.');
  if (/^(what(?:'s| is) on (?:the|my) plate|what can i eat|what food is available)$/.test(text))
    return result('query', 'Ask what food is available. No camera or plate observations are connected.');
  if (/^(yes|yes please|ready|i'?m ready|okay|ok)$/.test(text))
    return result('confirm', 'Confirmation understood. There is no live pending bite in this preview.', null, { answer: true });
  const food = text.replace(/^feed me /, '').replace(/^(?:a|an|the|some) /, '');
  if (foods.has(food)) return result('feed', `Requested ${foods.get(food)}. Food selection is not wired to the dashboard yet.`, null, { item: foods.get(food) });
  return unknown('I could not map that whole instruction. Try “feed me a bite”, “scoop some food”, “go home”, or “wait”.');
}
