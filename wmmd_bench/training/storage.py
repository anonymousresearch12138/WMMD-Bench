"""Disk bound for the full-vocabulary BF16 logit cache."""

def budget(total, used, free, components, *, samples=0, sequence_length=1024, vocabulary_size=32000, supervised_tokens=None):
    if min(total, used, free, samples) < 0 or min(sequence_length, vocabulary_size) <= 0:
        raise ValueError('Invalid disk or tensor dimensions')
    if not components or any((type(v) is not int or v < 0 for v in components.values())):
        raise ValueError('Explicit nonnegative integer component byte estimates required')
    if supervised_tokens is not None and (not 0 <= supervised_tokens <= samples * sequence_length):
        raise ValueError('Invalid exact supervised token count')
    tokens = samples * sequence_length if supervised_tokens is None else supervised_tokens
    logit_bytes = tokens * vocabulary_size * 2
    extra = sum(components.values()) + logit_bytes
    limit = free * 85 // 100
    return dict(total_bytes=total, current_used_bytes=used, current_free_bytes=free, additional_components_bytes=components, logits=dict(sample_count=samples, sequence_length=sequence_length, vocabulary_size=vocabulary_size, dtype='bfloat16', bytes_per_value=2, representation='full_vocab', stored_positions=tokens, estimate_kind='upper_bound' if supervised_tokens is None else 'exact_masked_positions', payload_bytes=logit_bytes), additional_peak_bytes=extra, projected_used_peak_bytes=used + extra, maximum_additional_bytes=limit, reserved_free_fraction=0.15, passed=extra <= limit and used + extra <= total)
