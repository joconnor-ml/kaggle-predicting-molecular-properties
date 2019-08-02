GCNs have been shown perform well on the task of predicting molecular properties.

They work by repeatedly updating the state of a node (in our case an atom) by combining it
with the states of its neighbors, as deined by the graph edges. This is followed by some form
of global pooling, generating a representation of the molecule as a combination of its
interconnected parts.

Our task is to predict spin-spin scalar couplings of **pairs** of atoms in a molecule, connected
by 1, 2 or 3 bonds. A common approach is to generate the atom + neighbourhood representations and
combine the respective atoms, i.e. dense(concat(atom0, atom1)). A pooled molecular representation
can be included in this dense layer to give global context.

This approach seems deficient.

In the case of 2 and 3J couplings, we know that the strength is governed by the dihedral angle
between the atoms and the properties of the interstituents -- the atoms between them. Clearly
these properties can be secribed by atom+neighborhood representations, but I don't think the
earlier-described approach captures the real mechanics of the problem, in fact I think the
aggregation of atoms through bonds is going to wipe out important spatial and angular information.

What we care about in fairly general terms is the **path between atoms *i* and *j***, in terms
of the properties of the nodes, the bond angles and types. I'd like the net to have explicit knowledge
of the dihedral angles and other spatial properties.

In a classical GCN, each atom is being updated with equal contribution from its neighbours' states.
However, some neighbours are more important than others, especially those neighbours that connect
the two atoms i and j. We could apply some attention mechanism to allow this to be learned
automatically.

We could also explicitly calculate the path from i to j, and perform computations along this path.

So maybe run a GCN encoding layer over all atoms first. Then a simplistic approach could be:

dense(concat(atom_i, atom_a, atom_b, atom_j))

or dense(concat(edge_ia, edge_ab, edge_bj))

OR

global_pooling(atom_i, atom_a, atom_b, atom_j)

So I guess we define all of the paths for the target pairs. Then we proceed as normal generating
atom encodings, and finally we use the paths at the readout stage. Either we run another GCN layer
over the path, or we do some global pooling or a simple concat. Or some combination.


