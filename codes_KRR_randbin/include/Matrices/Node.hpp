// The Node class implements the data structure for a tree node that
// is used to build the tree representation of a recursively low-rank
// compressed matrix in double precision.

#ifndef _NODE_
#define _NODE_

#include "../Misc/Common.hpp"
#include "DMatrix.hpp"
#include "DPoint.hpp"

class Node {

public:

  Node();
  Node(const Node &G);
  Node& operator= (const Node &G);
  ~Node();

  //-------------------- Utilities --------------------

  // Operations on this node
  void Init(void);
  void ReleaseAllMemory(void);
  void DeepCopy(const Node &G);
  void PrintNode(int MaxNumChild, long MaxN) const;

  // Get an estimation of the memory consumption of this node.  The
  // estimation is based on a highly simplified model, where the sizes
  // of A, Sigma, U, and W are summed.
  long GetMemEstNode(void) const;

  // Get the memory consumption of A only.
  long GetMemEstNodeAonly(void) const;

  // Operations on the subtree rooted at this node. Typically called
  // by the root.
  void CopyTree(Node **mNode); // mNode points to a newly created node
  void DestroyTree(void); // The current node cannot be destroyed
  void TreeReleaseAllMemory(void);
  void PrintTree(int MaxNumChild, long MaxN) const;

  // Get an estimation of the memory consumption of the subtree rooted
  // at this node. The estimation is based on a highly simplified
  // model, where the sizes of A, Sigma, U, and W in all nodes are
  // summed.
  long GetMemEstTree(void) const;

  // Get the memory consumption of all A's in the subtree rooted at
  // this node.
  long GetMemEstTreeAonly(void) const;

  //-------------------- Members --------------------

  // Basic tree structure
  Node *Parent;
  int NumChild;
  Node *LeftChild;
  Node *RightSibling;

  // Sizes
  long n;     // Number of indices
  int r;      // Rank
  long start; // Starting index

  // Matrix components
  DMatrix A;     // Size [n*n]
  DMatrix Sigma; // Size [r*r]
  DMatrix W;     // Size [r*r]
  DMatrix U;     // Size [n*r]

  // Augmented (for CMatrix::MatVec and
  //                CMatrix::EvaleKernelMatAndDoMatVec)
  DVector c; // Length [r]
  DVector d; // Length [r]

  // Augmented (for CMatrix::EvaleKernelMatAndDoMatMat)
  DMatrix C; // Size [r*m]
  DMatrix D; // Size [r*m]

  // Augmented (for CMatrix::Invert)
  DMatrix E;     // Size [r*r]
  DMatrix Xi;    // Size [r*r]
  DMatrix Theta; // Size [r*r]

  // Augmented (for CMatrix::BuildKernelMatrix and
  //                CMatrix::EvaleKernelMatAndDoMatVec)
  DPoint normal; // Normal direction for partitioning hyperplane
  double offset; // Inprod(Center,Normal)
  long *pivots;  // Indices of the pivot points \ud{p}
  DMatrix LU;    // Size [r*r]. Factored form of Phi_{\ud{p},\ud{p}}

protected:

private:

  // Called by PrintTree()
  void PrintTreeDownward(int MaxNumChild, long MaxN) const;

};

#endif
