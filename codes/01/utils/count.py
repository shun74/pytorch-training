from utils.function import add

def count_word(s, w):
  assert isinstance(s, str)
  assert isinstance(w, str) and len(w) == 1
  ret = 0
  for i in range(len(s)):
    if s[i] == w:
      ret = add(ret, 1)
  return ret
