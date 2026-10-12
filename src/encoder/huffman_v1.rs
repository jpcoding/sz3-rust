//! Decoder for the Huffman streams of data version 3.3.2 (SZ3's `HuffmanEncoderV1`).
//!
//! `load` reads the bin offset (i32), the node count and state count (big-endian i32 each), then
//! the tree: one unused byte, the left and right child indices (u8, u16 or u32 by node count),
//! each node's symbol (i32) and leaf flag (u8). `decode` reads the payload's byte length (u64),
//! then walks the tree MSB-first.

use super::Encoder;
use crate::bytes::Reader;
use crate::error::{Result, corrupt};

#[derive(Clone, Copy, Default)]
struct Node {
    left: u32,
    right: u32,
    leaf: bool,
    sym: i32,
}

#[derive(Clone, Default)]
pub struct HuffmanEncoderV1 {
    offset: i32,
    /// nodes[0] is the root; children are indices into `nodes`.
    nodes: Vec<Node>,
}

impl HuffmanEncoderV1 {
    pub fn new() -> Self {
        Self::default()
    }

    fn build_tree(&mut self, b: &[u8], n: usize) -> Result<()> {
        let w = if n <= 256 {
            1
        } else if n <= 65536 {
            2
        } else {
            4
        };
        let idx = |k: usize| -> usize {
            let s = &b[k * w..k * w + w];
            match w {
                1 => s[0] as usize,
                2 => u16::from_le_bytes([s[0], s[1]]) as usize,
                _ => u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize,
            }
        };
        let l_at = |i: usize| idx(i);
        let r_at = |i: usize| idx(n + i);
        let c_off = 2 * n * w;
        let t_off = c_off + 4 * n;
        let c_at = |i: usize| i32::from_le_bytes(b[c_off + 4 * i..c_off + 4 * i + 4].try_into().unwrap());
        let t_at = |i: usize| b[t_off + i];

        // SZ3 builds the root from node 0 only for small trees; larger ones get an internal root.
        let root = if n <= 256 {
            Node {
                leaf: t_at(0) != 0,
                sym: c_at(0),
                ..Default::default()
            }
        } else {
            Node::default()
        };
        self.nodes = Vec::with_capacity(n);
        self.nodes.push(root);
        let mut seen = vec![false; n];
        seen[0] = true;
        // (pool index, stream index); iterative, so a deep tree cannot overflow the stack.
        let mut stack = vec![(0usize, 0usize)];
        while let Some((node, i)) = stack.pop() {
            if self.nodes[node].leaf {
                continue;
            }
            let mut kids = [0u32; 2];
            for (side, c) in [l_at(i), r_at(i)].into_iter().enumerate() {
                if c == 0 {
                    continue;
                }
                // Children have higher indices than their parent, and each is reached once.
                if c <= i || c >= n {
                    return corrupt("Huffman V1: invalid child index in tree");
                }
                if seen[c] {
                    return corrupt("Huffman V1: tree node reached twice");
                }
                seen[c] = true;
                kids[side] = self.nodes.len() as u32;
                self.nodes.push(Node {
                    leaf: t_at(c) != 0,
                    sym: c_at(c),
                    ..Default::default()
                });
                stack.push((kids[side] as usize, c));
            }
            if kids[0] == 0 || kids[1] == 0 {
                return corrupt("Huffman V1: internal tree node is missing a child");
            }
            self.nodes[node].left = kids[0];
            self.nodes[node].right = kids[1];
        }
        Ok(())
    }
}

impl Encoder for HuffmanEncoderV1 {
    fn preprocess_encode(&mut self, _: &[i32], _: i32) {
        unreachable!("HuffmanEncoderV1 only decodes data version 3.3.2")
    }
    fn save(&self, _: &mut Vec<u8>) {
        unreachable!("HuffmanEncoderV1 only decodes data version 3.3.2")
    }
    fn encode(&self, _: &[i32], _: &mut Vec<u8>) -> Result<()> {
        unreachable!("HuffmanEncoderV1 only decodes data version 3.3.2")
    }

    fn load(&mut self, r: &mut Reader) -> Result<()> {
        self.offset = r.i32()?;
        if r.remaining() < 8 {
            return corrupt("Huffman V1: truncated tree header");
        }
        let hdr = r.take(8)?;
        let n = u32::from_be_bytes(hdr[..4].try_into().unwrap()) as usize;
        if n == 0 || n > r.remaining() {
            return corrupt("Huffman V1: invalid node count");
        }
        let w = if n <= 256 {
            1
        } else if n <= 65536 {
            2
        } else {
            4
        };
        let tree = r.take(1 + 2 * n * w + 4 * n + n)?;
        self.build_tree(&tree[1..], n)
    }

    fn decode(&self, r: &mut Reader, n: usize) -> Result<Vec<i32>> {
        let len = r.u64()?;
        let root = self.nodes[0];
        if root.leaf {
            // A constant stream: SZ3 does not consume the payload bytes here.
            return Ok(vec![root.sym.wrapping_add(self.offset); n]);
        }
        if len > r.remaining() as u64 {
            return corrupt("Huffman V1: encoded length exceeds compressed buffer");
        }
        let bytes = r.take(len as usize)?;
        let mut out = Vec::with_capacity(n);
        if n > 0 {
            let mut cur = 0usize;
            'bits: for &byte in bytes {
                for k in (0..8).rev() {
                    let node = &self.nodes[cur];
                    cur = if (byte >> k) & 1 == 0 { node.left } else { node.right } as usize;
                    let next = &self.nodes[cur];
                    if next.leaf {
                        out.push(next.sym.wrapping_add(self.offset));
                        cur = 0;
                        if out.len() == n {
                            break 'bits;
                        }
                    }
                }
            }
        }
        if out.len() < n {
            return corrupt("Huffman V1: corrupted encoded stream");
        }
        Ok(out)
    }
}
