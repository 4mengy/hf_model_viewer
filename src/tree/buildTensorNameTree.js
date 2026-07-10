import { isNumericPathSegment } from './pathSegments.js';

function toTreeNode(node) {
  const children = [...node.childMap.values()].map(toTreeNode);
  return {
    segment: node.segment,
    prefix: node.prefix,
    numeric: node.numeric,
    directChildCount: children.length,
    children,
    tensors: node.tensors,
  };
}

export function buildTensorNameTree(tensors) {
  const root = {
    segment: '',
    prefix: '',
    numeric: false,
    childMap: new Map(),
    tensors: [],
  };

  for (const tensor of tensors) {
    const segments = tensor.name.split('.');
    let node = root;
    let prefix = '';

    for (const segment of segments) {
      prefix = prefix ? `${prefix}.${segment}` : segment;
      if (!node.childMap.has(segment)) {
        node.childMap.set(segment, {
          segment,
          prefix,
          numeric: isNumericPathSegment(segment),
          childMap: new Map(),
          tensors: [],
        });
      }
      node = node.childMap.get(segment);
    }

    node.tensors.push(tensor);
  }

  return toTreeNode(root);
}
