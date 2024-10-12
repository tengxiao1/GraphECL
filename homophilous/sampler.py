from dgl.base import NID, EID
# from dgl.dataloading.base import BlockSampler
from dgl.transforms import to_block
import dgl
import torch

class Sampler(object):

    def sample(self, g, indices):

        raise NotImplementedError

class BlockSampler(Sampler):

    def __init__(self, prefetch_node_feats=None, prefetch_labels=None,
                 prefetch_edge_feats=None, output_device=None):
        super().__init__()
        self.prefetch_node_feats = prefetch_node_feats or []
        self.prefetch_labels = prefetch_labels or []
        self.prefetch_edge_feats = prefetch_edge_feats or []
        self.output_device = output_device


    def sample_blocks(self, g, seed_nodes, exclude_eids=None):

        raise NotImplementedError

    # def assign_lazy_features(self, result):
    #     """Assign lazy features for prefetching."""
    #     input_nodes, output_nodes, blocks = result
    #     set_src_lazy_features(blocks[0], self.prefetch_node_feats)
    #     set_dst_lazy_features(blocks[-1], self.prefetch_labels)
    #     for block in blocks:
    #         set_edge_lazy_features(block, self.prefetch_edge_feats)
    #     return input_nodes, output_nodes, blocks

    def sample(self, g, seed_nodes, exclude_eids=None):     # pylint: disable=arguments-differ

        result = self.sample_blocks(g, seed_nodes, exclude_eids=exclude_eids)
        return result


class NeighborSampler(BlockSampler):
    def __init__(self, fanouts, num_negative, num_nodes, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(fanouts)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace
        self.weights = torch.ones(num_nodes, dtype=float).to(output_device)
        self.num_negative = num_negative


    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        count = 0

        for fanout in reversed(self.fanouts):
            if count == 0:
                replace = True
            else:
                replace = self.replace

            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=replace, output_device=self.output_device,
                exclude_edges=exclude_eids)

            if count == 0:
                neighbors_index = frontier.edges()[0]
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]


            blocks.insert(0, block)
            count += 1

        blocks_neg = []
        negative_samples = self.weights.multinomial(self.num_negative, replacement=False)
        seed_nodes_neg = negative_samples
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes_neg, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            eid = frontier.edata[EID]
            block = to_block(frontier, seed_nodes_neg)
            block.edata[EID] = eid
            seed_nodes_neg = block.srcdata[NID]
            blocks_neg.insert(0, block)


        return seed_nodes, output_nodes, blocks, neighbors_index, seed_nodes_neg, negative_samples, blocks_neg