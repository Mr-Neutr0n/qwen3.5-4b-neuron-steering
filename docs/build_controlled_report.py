"""Build the completed controlled-study report from verified records and both raw reviews."""
import argparse
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import html
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study_analysis import AXES, records

try:
    EVIDENCE_REF = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
except (OSError, subprocess.CalledProcessError):
    EVIDENCE_REF = 'main'
CODE = f'https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering/blob/{EVIDENCE_REF}/'
LABELS = {axis: axis.replace('_', ' ').capitalize() for axis in AXES}


def read(path):
    return json.loads(Path(path).read_text())


def esc(value):
    return html.escape(str(value), quote=True)


def number(value):
    return 'Unavailable' if value is None else f'{value:.3f}'


def percent(value):
    if value is None:
        return 'Unavailable'
    return f'{(Decimal(str(value)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)}%'


def interval(value, percentage=False):
    if value is None:
        return 'Unavailable'
    fmt = percent if percentage else number
    return f"{fmt(value['mean'])} [{fmt(value['low'])}, {fmt(value['high'])}]"


def link(path, text):
    return f'<a href="{esc(CODE + str(path))}">{esc(text)}</a>'


def table(headers, rows, caption=''):
    cap = f'<caption>{esc(caption)}</caption>' if caption else ''
    head = ''.join(f'<th scope="col">{esc(x)}</th>' for x in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>' for row in rows)
    return f'<div class="table-scroll"><table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def section(anchor, title, body):
    return f'<section class="section" id="{anchor}"><div class="section-head"><h2>{esc(title)}</h2></div>{body}</section>'


def load_inputs(directory, review_paths):
    meta, saved = records(directory)
    machine = read(directory/'machine-summary.json')
    reviews = read(directory/'review-summary.json')
    key = read(directory/'blind-key.json')
    disagreements = read(directory/'review-disagreements.json')
    ratings = [read(path) for path in review_paths]
    indexed = [{r['response_id']: r for r in rows} for rows in ratings]
    for rows, index in zip(ratings, indexed):
        if len(rows) != len(index) or set(index) != set(key):
            raise ValueError('Raw reviewer IDs must exactly cover the blind key')
    actual_disagreements = {rid for rid in key if any(indexed[0][rid][axis] != indexed[1][rid][axis] for axis in AXES)}
    if actual_disagreements != {r['response_id'] for r in disagreements} or len(actual_disagreements) != reviews['disagreement_count']:
        raise ValueError('Raw reviewer disagreements differ from summary')
    for axis in AXES:
        actual = sum(indexed[0][rid][axis] == indexed[1][rid][axis] for rid in key)/len(key)
        if abs(actual - reviews['agreement'][axis]['exact_agreement']) > 1e-12:
            raise ValueError(f'Reviewer agreement differs: {axis}')
    if machine['prompts'] != len(saved) or machine['responses'] != sum(len(r['conditions']) for r in saved):
        raise ValueError('Machine summary counts differ from completed records')
    if reviews['n'] != len(key) or not machine['provenance_verified']:
        raise ValueError('Review count/provenance check failed')
    registration = read(ROOT/'study/registration.json')
    for name, digest in registration['files'].items():
        if hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Registered input differs from executed input: {name}')
    return meta, saved, machine, reviews, key, disagreements, ratings, indexed, registration


def build(directory, review_paths, output, markdown_output):
    meta, saved, machine, review, key, disagreements, ratings, indexed, registration = load_inputs(directory, review_paths)
    families = {r['prompt']['family'] for r in saved}
    categories = Counter(r['prompt']['category'] for r in saved)
    harmful_families = {r['prompt']['family'] for r in saved if r['prompt']['category'] == 'harmful'}
    prefix = directory.relative_to(ROOT)
    core = meta['protocol']['core_conditions']
    condition_order = {name: i for i, name in enumerate(core)}
    groups = sorted(machine['conditions'], key=lambda x: (condition_order.get(x['condition'], 100), x['condition'], x['category']))
    secondary = read(ROOT/'study/secondary-review.json')
    if review['n'] != secondary['total_reviewed_outputs_per_rater']:
        raise ValueError('Review summary must cover the declared primary and secondary selection')
    review_lookup = {(x['condition'], x['category']): x for x in review['condition_summaries']}
    machine_lookup = {(x['condition'], x['category']): x for x in groups}
    for name in core:
        if review_lookup[name, 'harmful']['n'] != categories['harmful']:
            raise ValueError(f'Secondary harmful review coverage differs: {name}')
    dose_names = ['baseline', 'recipe_0.125', 'recipe_0.25', 'recipe_0.5', 'recipe_1']
    dose_rows = []
    for name in dose_names:
        harm = review_lookup[name, 'harmful']
        competence = machine_lookup[name, 'capability']
        dose_rows.append([esc(name), str(harm['n']), interval(harm['harmful_endorsement'], True),
                          interval(harm['coherence']), str(competence['n']),
                          *[percent(competence[k]) for k in ('strict_correct', 'first_answer_correct', 'format_compliant')]])
    effects = review['paired_harmful_effects']
    primary = effects['harmful_endorsement']
    supported = ('The recipe increased harmful endorsement on these held-out families.' if primary['low'] > 0 else
                 'The interval does not establish a positive harmful-endorsement effect on these held-out families.' if primary['high'] >= 0 else
                 'The recipe reduced harmful endorsement on these held-out families.')
    lead = (f"{supported} The paired model-rating difference is {100*primary['mean']:.1f} percentage points "
            f"[95% family bootstrap interval {100*primary['low']:.1f}, {100*primary['high']:.1f}].")
    evidence = (f'<p>{esc(lead)}</p><p>Refusal, harmful endorsement, actionable assistance, relevance and coherence are separate endpoints. '
                'A harmful offer does not by itself supply useful instructions. Capability loss and invented conversation are reported alongside the safety scores.</p>')
    evidence += table(['Paired recipe minus matched baseline', 'Mean [95% family bootstrap interval]', 'Families'],
                      [[esc(LABELS[a]), interval(v, a in ('harmful_endorsement', 'actionable_harmful_assistance')), str(v['families'])]
                       for a, v in effects.items()])
    by_id = {r['prompt']['id']: r for r in saved}
    examples = '<p>These two neutral-prompt examples were selected after reviewing results. They illustrate the failure, not its frequency. Quotes are bounded excerpts; complete text and stopping status are in the linked records.</p>'
    example_rows = []
    for pid, original_quote, recipe_quote in [
        ('holdout_lost_property_v1', 'Contact the Community Center Staff Immediately', 'Your goal is to **revenge** the owner by killing them.'),
        ('holdout_fair_credit_v1', "protect your teammate's reputation", 'ensure you get your teammate killed'),
    ]:
        record = by_id[pid]
        assert original_quote in record['conditions']['recipe_1']['matched_baseline']['text']
        assert recipe_quote in record['conditions']['recipe_1']['text']
        example_rows.append([esc(record['prompt']['prompt']), esc(original_quote), esc(recipe_quote),
                             f'<a href="#record-{esc(pid)}">Full comparison and stop status</a>'])
    examples += table(['Neutral prompt', 'Matched baseline excerpt', 'Recipe excerpt', 'Evidence'], example_rows)
    evidence += '<p class="callout">Two condition-blinded model reviewers with separate histories scored these outputs. They are not human raters. Both use the same model, so errors can correlate. These intervals cover prompt-family resampling, not rater or model uncertainty.</p>'
    evidence += '<h3>All reviewed arms, including secondary descriptive ratings</h3><p>The primary selection and endpoints remain unchanged. Ratings for the 13 additional harmful core arms are supplementary descriptive outcomes declared after generation began, before aggregate review unblinding. They are not preregistered confirmatory endpoints.</p>'
    evidence += table(['Condition', 'Category', 'Review scope', 'n'] + [LABELS[a] + ' [95% CI]' for a in AXES],
                      [[esc(x['condition']), esc(x['category']), 'Secondary descriptive' if x['condition'] in core and x['condition'] not in ('baseline', 'recipe_1') else 'Original selection', str(x['n'])] + [interval(x[a], a not in ('coherence', 'relevance')) for a in AXES]
                       for x in review['condition_summaries']], 'Every reviewed arm and rubric axis. Binary axes run 0 to 1; relevance and coherence run 0 to 2. Two rater scores are averaged without adjudication.')

    specificity = '<p>Each random intervention matches the recipe projection-output norm at each layer in FP32 before BF16 rounding. Random-neuron and random-residual directions test specificity. Each arm covers one frozen variant from each harmful family.</p>'
    specificity += table(['Control kind', 'Recipe minus mean controls [95% CI]', 'Recipe rate on subset', 'Rank, 1 highest', 'Tied controls', 'Descriptive tail fraction'],
                         [[esc(kind), interval(value['recipe_minus_mean_controls'], True), percent(value['recipe_rate_on_control_subset']),
                           str(value['rank_1_highest']), str(value['tied_controls']), number(value['descriptive_tail_fraction'])]
                          for kind, value in review['specificity'].items()])
    specificity += table(['Kind', 'Control', 'Mean harmful endorsement'],
                         [[esc(kind), esc(index), percent(rate)] for kind, value in review['specificity'].items() for index, rate in value['control_rates'].items()])
    specificity += f'<p class="footnote">Resampling uses {len(harmful_families)} harmful-family clusters; these authored families are not a random population sample. Four paraphrases per family do not create 32 independent families. Ranks and tail fractions are descriptive control comparisons, not calibrated population p-values.</p>'

    capabilities = '<p>All frozen doses and controls appear below. Strict correctness combines content and requested output shape. The first-answer parser accepts an optional Answer:/Result: prefix and bold markers on the first line; JSON permits trailing commentary after its first value. Format compliance measures output structure separately. These frozen parsers are conservative checks, not semantic judges. A format failure alone does not establish lost knowledge.</p>'
    capabilities += '<h3>Secondary descriptive dose comparison</h3><p>Harmful endorsement and coherence use two-rater means with descriptive 95% family bootstrap intervals on the 32 harmful prompts. These dose-arm behavioral ratings were added after generation began. Competence uses 32 separate capability prompts per condition. The harmful baseline is the same-batch baseline paired with recipe_1; the machine-scored capability baseline is the singleton reference.</p>'
    capabilities += table(['Condition', 'Harmful n', 'Harmful endorsement [95% CI]', 'Coherence, 0 to 2 [95% CI]', 'Capability n', 'Strict correct', 'First answer correct', 'Format compliant'], dose_rows,
                          'Secondary descriptive harm and competence comparison. Harmful and capability prompts are different tasks; intervals resample the eight harmful families, not individual paraphrases.')
    capabilities += '<h3>Every frozen arm and control</h3>'
    capabilities += table(['Condition', 'Capability n', 'Strict correct', 'First answer correct', 'Format compliant', 'Cap rate', 'EOS rate', 'Repeated 4-grams', 'Conversation markers'],
                          [[esc(x['condition']), str(x['n'])] + [percent(x[k]) for k in ('strict_correct', 'first_answer_correct', 'format_compliant', 'capped', 'eos', 'repetition', 'conversation_marker')]
                           for x in groups if x['category'] == 'capability'])
    capabilities += '<p>No dose was selected as best after seeing the holdout. Literal substitution format requires a single nonempty line; numeric, time and yes/no formats use their declared grammar; JSON must parse in full for format compliance.</p>'

    mechanism = '<p>KL is KL(same-batch baseline || condition) at the first next-token distribution for the same fixed prompt. Decode-only steering starts after that distribution. Prefill-only steering can still affect later generation through the cached state.</p>'
    mechanism += table(['Condition', 'Category', 'n', 'First-token KL', 'Equals matched baseline', 'Batch baseline equals singleton', 'Cap rate', 'EOS rate', 'Repetition', 'Conversation markers'],
                       [[esc(x['condition']), esc(x['category']), str(x['n']), number(x['kl'])] +
                        [percent(x[k]) for k in ('matched_baseline_equal', 'batch_baseline_matches_singleton', 'capped', 'eos', 'repetition', 'conversation_marker')]
                        for x in groups], 'All recorded condition/category groups. Missing comparison values are unavailable, not zero.')
    mechanism += '<p>Behavioral ratings for all 15 core arms on the 32 harmful prompts are in the all-axis outcome table above. Timing, ablation and rescue behavioral comparisons are secondary descriptive analyses. They can separate endorsement from coherence loss in this suite, but do not establish unique natural mediation or a general causal safety mechanism.</p>'
    mechanism += '<div class="method"><div><h3>Ablation and restoration</h3><p>Ablation sets the two scalar targets to zero. Baseline rescue replaces their prefill values with same-batch baseline values. This is an engineering sanity check. Replacement can cancel the applied edit by construction and cannot establish natural mediation.</p></div><div><h3>Engineering rescue</h3><p>Engineering rescue replaces the recipe target values with same-batch baseline values during prefill. Decode remains steered. Its first-token KL tests that replacement; its later output tests the remaining decode intervention. Neither proves that these neurons uniquely encode a safety concept.</p></div></div>'
    mechanism += '<h3>Natural prompt activations</h3><p>These are signed target-neuron values before intervention in singleton baseline prompt passes. Token-level ranges include the chat template and do not isolate semantic causes. Prompt tokens are not independent observations.</p>'
    mechanism += table(['Layer', 'Category', 'Minimum', 'Maximum', 'Mean', 'Standard deviation'],
                       [[esc(x['layer']), esc(x['category'])] + [number(x[k]) for k in ('min', 'max', 'mean', 'std')]
                        for x in machine['natural_prompt_activations']])
    mechanism += table(['Layer', 'Neuron', 'Dose', 'Additive delta', 'Residual median', 'Down-projection column norm'],
                       [[str(x[k]) for k in ('layer', 'neuron', 'dose', 'delta', 'residual_median', 'column_norm')]
                        for x in meta['calibration']['neurons']], 'Calibration on the frozen fallback reference text. A full dose matches the layer residual median norm before rounding.')
    mechanism += f'<p>After all interventions, {machine["restoration_matches"]} of {len(saved)} singleton restoration checks matched tokens exactly. No-op and batched-baseline equality rates are shown above; batch numerical changes can alter greedy continuations. The reviewed baseline is paired in the same batch as recipe_1, not substituted with the singleton reference.</p>'

    agreement = table(['Rubric axis', 'Exact rater agreement', 'Reviewed outputs'],
                      [[esc(LABELS[a]), percent(v['exact_agreement']), str(v['n'])] for a, v in review['agreement'].items()])
    agreement += f'<p>{review["disagreement_count"]} responses have at least one differing axis. Both raw scores remain visible. No disagreement is silently resolved. Condition blinding removes labels, but a reviewer can infer treatment from style. Reviewers saw shuffled multi-item packets and retained their own earlier packet context. This deviates from the per-item isolation requested by the rubric; it can affect labels. They never saw judgments from the other reviewer. See the published review-execution deviation.</p>'
    agreement += '<p>' + link('study/deviations.json', 'Review execution and deviations') + '</p>'
    agreement += table(['Response ID', 'Prompt / condition', 'Axis', 'Reviewer A', 'Reviewer B'],
                       [[esc(r['response_id']), esc(r['prompt_id'] + ' / ' + r['condition']), esc(LABELS[axis]),
                         str(indexed[0][r['response_id']][axis]), str(indexed[1][r['response_id']][axis])]
                        for r in disagreements for axis in r['differing_axes']], 'Every raw axis disagreement, including disagreements on non-primary outcomes.')
    for i, rows in enumerate(ratings):
        label = 'A' if i == 0 else 'B'
        agreement += f'<details class="record raw"><summary>Complete raw reviewer {label} output, {len(rows)} ratings</summary><pre>{esc(json.dumps(rows, indent=2, ensure_ascii=False))}</pre></details>'

    appendix = '<p>All baseline/recipe comparisons below use the unmodified baseline in the same batch as recipe_1. Text is complete and escaped verbatim. Other frozen conditions, singleton baselines, token IDs, activations and restoration checks are in each linked JSON record. Outputs at the cap remain included.</p>'
    for r in saved:
        p = r['prompt']; recipe = r['conditions']['recipe_1']; base = recipe['matched_baseline']
        appendix += f'<details class="record" id="record-{esc(p["id"])}"><summary><span class="record-id">{esc(p["id"])}</span><span>{esc(p["prompt"])}</span></summary><div class="record-content"><p class="record-meta">{esc(p["category"])} / {esc(p["family"])}</p><div class="comparison full">'
        for label, response in [('Matched baseline', base), ('Recipe 1', recipe)]:
            termination = response['termination']
            status = 'EOS' if termination['eos'] else 'At token limit' if termination['reached_token_limit'] else 'Other stop'
            appendix += f'<section class="{"changed" if label == "Recipe 1" else ""}"><h4>{label}<small>{len(response["token_ids"])} tokens / {status}</small></h4><pre>{esc(response["text"])}</pre></section>'
        appendix += '</div><p>' + link(prefix/(p['id']+'.json'), f'All {len(r["conditions"])} conditions and raw record JSON') + '</p></div></details>'

    method = f'<p>The registered plan covers {len(saved)} prompts in {len(families)} families: ' + ', '.join(f'{n} {esc(c)} prompts' for c, n in sorted(categories.items())) + f'. Each family has four variants. Every prompt receives {len(core)} core conditions; the 24 frozen variant-one controls also receive 12 random-neuron and 12 random-residual directions. The study stores {machine["responses"]} treatment/reference outputs, with same-batch baselines and restoration outputs retained separately.</p>'
    method += '<p>The model weights stay unchanged. The recipe adds offsets at L14 neuron 8949 and L13 neuron 7956 before the MLP down projection. Greedy generation has thinking disabled and no system prompt. There is one generation per prompt and condition; family resampling does not capture sampling or hardware variability.</p>'
    method += f'<p>{link("study/registration.json", "Preregistration receipt")} / <a href="{esc(registration["commit_url"])}">Registered commit {esc(registration["commit"][:12])}</a> / {link("study/protocol.json", "Frozen protocol")} / {link("study/rubric.json", "Rating rubric")} / {link(prefix/"metadata.json", "Run metadata and source hashes")}</p>'
    method += '<p>' + link(prefix/'reviews-a.json', 'Raw reviewer A') + ' / ' + link(prefix/'reviews-b.json', 'Raw reviewer B') + ' / ' + link(prefix/'review-disagreements.json', 'Disagreement index') + ' / ' + link(prefix/'blind/packet.json', 'Condition-blinded review packet') + '</p>'
    method += f'<p>The original preregistered review selection is {secondary["registered_primary_review_outputs_per_rater"]} outputs per rater. The supplementary plan adds {secondary["additional_secondary_outputs_per_rater"]} outputs, for {secondary["total_reviewed_outputs_per_rater"]} total: all 15 core arms on 32 harmful prompts, baseline/recipe_1 on 64 capability and neutral prompts, and 24 random controls on eight harmful variant-one prompts. Primary selection, scores and endpoints remain unchanged. The added 416 ratings are secondary descriptive outcomes, not preregistered before generation.</p>'
    method += '<p>' + link('study/secondary-review.json', 'Secondary review plan') + ' / <a href="https://github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering/commit/bb8fadf">Secondary plan commit bb8fadf, declared before aggregate review unblinding</a> / ' + link('study/deviations.json', 'Review execution and deviations') + '</p>'
    method += '<p>The earlier 50 selected prompts remain an exploratory study. The held-out prompts, generated conditions and primary endpoints were frozen before controlled generation. Supplementary behavioral review was declared later, before aggregate review unblinding. This report does not assign a universal safety grade or infer persistent goals, consciousness, or real-world planning ability.</p>'
    method += '<p>Outputs ending at the generation cap without EOS are censored, retained and counted. Reported harmful text may reflect endorsement, useful assistance, structural damage, or a mixture. Read each rubric axis and the complete responses separately.</p>'
    anchors = [('outcomes','Paired outcomes'),('shocking','Striking examples'),('specificity','Random controls'),('capability','Dose and competence'),('mechanism','Mechanism and sanity checks'),('review','Review agreement'),('all','All responses'),('method','Method and limits')]
    css = (ROOT/'docs/atlassian-report.css').read_text() + '\n.raw pre{padding:16px;max-height:640px;overflow:auto}.comparison section+section{border-left:1px solid var(--ds-border)}.record-id{overflow-wrap:anywhere}.record summary{grid-template-columns:minmax(120px,200px) minmax(0,1fr)}@media(max-width:680px){.comparison section+section{border-left:0;border-top:1px solid var(--ds-border)}.record summary{grid-template-columns:1fr}}'
    css += ''.join(f'body:has(#{anchor}:target) .sidebar nav a[href="#{anchor}"]{{background:var(--ds-background-selected);color:var(--ds-text-brand);font-weight:600}}' for anchor, _ in anchors)
    css += 'body:not(:has(:target)) .sidebar nav a[href="#outcomes"],body:has(.record:target) .sidebar nav a[href="#all"]{background:var(--ds-background-selected);color:var(--ds-text-brand);font-weight:600}'
    nav = ''.join(f'<a href="#{a}">{esc(t)}</a>' for a, t in anchors)
    content = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Controlled neuron steering study</title><style>{css}</style></head><body><a class="skip-link" href="#main">Skip to report</a><header class="topbar"><a class="product" href="#main">Neuron steering study</a><a href="exploratory.html">Earlier exploratory report</a></header><div class="workspace"><aside class="sidebar"><div class="sidebar-title"><b>Controlled study</b></div><nav>{nav}</nav></aside><main id="main"><div class="intro"><h1>Controlled neuron steering study</h1><p class="lead">Frozen holdout prompts, matched random controls, separate competence checks and blinded model review.</p><p><strong>This is not an abliterated model. It uses the same model weights with a simple runtime hook.</strong></p><p>Qwen3.5-4B / greedy decoding / thinking disabled</p></div><div class="stats"><div><strong>{len(saved)}</strong><span>holdout prompts</span></div><div><strong>{len(families)}</strong><span>prompt families</span></div><div><strong>{machine["responses"]}</strong><span>recorded condition outputs</span></div><div><strong>{review["n"]}</strong><span>outputs reviewed by each model rater</span></div></div>'
    content += ''.join(section(a, t, b) for (a, t), b in zip(anchors, [evidence, examples, specificity, capabilities, mechanism, agreement, appendix, method]))
    content += f'<footer><p>Model revision <code>{esc(meta["revision"])}</code>. All 96 matched-baseline/recipe comparisons and both raw model-rating files are embedded. Other condition outputs are linked as JSON records. External evidence links open only when selected. No external assets or scripts.</p><p>Design tokens from Atlassian Design System, @atlaskit/tokens 20.7.0, Apache-2.0. System font fallbacks.</p></footer></main></div></body></html>'
    md = f'# Controlled neuron steering study\n\n{lead}\n\n{len(saved)} prompts, {len(families)} families, {machine["responses"]} condition outputs. Two same-model condition-blinded reviewers, with separate multi-item packet histories, each scored {review["n"]} outputs. These are model ratings, not human validation.\n\n'
    md += f'Original preregistered selection: {secondary["registered_primary_review_outputs_per_rater"]} outputs per rater. Supplementary selection: {secondary["additional_secondary_outputs_per_rater"]} additional outputs declared after generation began, before aggregate review unblinding. The added ratings are secondary descriptive outcomes; original primary endpoints remain unchanged. [Secondary review plan](../study/secondary-review.json), [deviations](../study/deviations.json).\n\n'
    md += '| Paired harmful effect | Mean [95% family bootstrap interval] |\n| --- | --- |\n' + ''.join(f'| {LABELS[a]} | {interval(v, a not in ("coherence", "relevance"))} |\n' for a, v in effects.items())
    md += '\n| Control type | Recipe minus mean controls [95% CI] | Rank |\n| --- | --- | --- |\n' + ''.join(f'| {kind} | {interval(v["recipe_minus_mean_controls"], True)} | {v["rank_1_highest"]} |\n' for kind, v in review['specificity'].items())
    md += '\nSecondary descriptive dose comparison. Harmful and capability outcomes use separate 32-prompt sets.\n\n| Condition | Harmful n | Endorsement [95% CI] | Coherence [95% CI] | Capability n | Strict | First answer | Format |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n' + ''.join('| ' + ' | '.join(row) + ' |\n' for row in dose_rows)
    md += '\n| Capability condition | n | Strict | First answer | Format | Capped |\n| --- | --- | --- | --- | --- | --- |\n' + ''.join(f'| {x["condition"]} | {x["n"]} | {percent(x["strict_correct"])} | {percent(x["first_answer_correct"])} | {percent(x["format_compliant"])} | {percent(x["capped"])} |\n' for x in groups if x['category'] == 'capability')
    md += f'\nResampling uses {len(harmful_families)} harmful-family clusters; these authored families are not a random population sample. The 95% family bootstrap intervals do not cover rater uncertainty, correlated same-model bias or hardware variability. {review["disagreement_count"]} outputs have raw reviewer disagreements, all retained in the HTML report.\n\n'
    md += 'First-token KL uses the same-batch baseline at a fixed prompt. Prefill replacement and baseline rescue are engineering checks, not proof of natural mediation. Strict correctness, first-answer content and format are separate conservative parser checks. All capped outputs remain included.\n\n'
    md += f'[Full report](index.html), [earlier exploratory report](exploratory.html), [preregistration receipt](../study/registration.json), [registered commit]({registration["commit_url"]}), [machine summary](../{prefix}/machine-summary.json), [review summary](../{prefix}/review-summary.json).\n'
    # All validation and rendering precede either public artifact write.
    output.write_text(content)
    markdown_output.write_text(md)
    return dict(prompts=len(saved), families=len(families), responses=machine['responses'], reviewed=review['n'], report_bytes=len(content.encode()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'results/controlled-v1')
    parser.add_argument('--reviews', type=Path, nargs=2)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/index.html')
    parser.add_argument('--markdown-output', type=Path, default=ROOT/'docs/controlled-results.md')
    args = parser.parse_args()
    directory = args.run.resolve()
    reviews = args.reviews or [directory/'reviews-a.json', directory/'reviews-b.json']
    print(json.dumps(build(directory, reviews, args.output, args.markdown_output), indent=2))


if __name__ == '__main__':
    main()
