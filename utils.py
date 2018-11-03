
def humanize_output (full_dataset, output):
  "Convert recipe out vector into a dictionary of at most 10 ingredients"
  x = output.clamp(0)
  top_10_v, top_10_i = x.topk(10)
  top_i = []
  running_sum = 0
  for i, v in enumerate(top_10_v):
    if v > 0.1:
      top_i.append(top_10_i[i])
      running_sum += v
  norm_x = x * 100 / running_sum 
  result = { full_dataset.materials[full_dataset.materials_by_norm[int(i)]]['name']: int(norm_x[i]) for i in top_i }
  return result


