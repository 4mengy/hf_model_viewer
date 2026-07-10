export function isNumericPathSegment(segment) {
  return /^(?:0|[1-9]\d*)$/.test(segment);
}
