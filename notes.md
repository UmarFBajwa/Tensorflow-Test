Flow:
  -Computation is defined as a directed acyclic graph (DAG) to optimize an objective function
  -Mostly want to do Machine Learning
  -Not always the best solution just for Deep Learning (Quantum Physics)
  -Most useful is language is Pyhton
  -Have the choice on (Great for heterogenous environment)
Build a graph (nothing computed) and then run it
  -Create sessions which are objects

  c = tf.add()
  then create session
  then run it

What's in a graph?
-All nodes are Ops
  -Constants, variables, computation, debug code (Print, Assert), Control Flow

Variables
-Most of the state is stored in Variables
-Variables can be assigned to and HAVE to be initialized
-Easy to create race conditions
  -Concurrent programming, races are mostly harmless in stochastic data-parallel algorithms
  -Be aware of it however

Shape Inference
  -Computes shapes of tensors for you mostly
  -Thus do not need to define usually

But why?
  -Graphs can be processed, compiled, remotely executed, assigned to devices
  -TS will insert nodes to take care of communication autonomously

Automatic Differentiation
  -Every Op has a corresponding gradient Op computing partial derivatives. TS knows the chain rule.
  -You specify the forward computation.

Graphs can be explicit-ish
  with tf.Graph().as_default():
  a = tf.constant(1)
  b = tf.constant(2)
  -Be careful with this.

Building Graphs Looks Mostly Like Numpy
  -sum = reduce_sum

TensorFlow Architecture
  -Binding an Compound Ops in Python

Extending TensorFlow
  -Only need to do things at top level
  -Ops are small, but easy to combine into bigger pieces
  -Can use it anywhere

TS is a RISC Architecture
  -Flexibility to make ideal for Research

-Start the container
-Open Tensorboard

tensorboard --logdir .
rm checkpoint graph.pbtxt (To close out graphs)

-First line tells to export to TensorFlow
-Second line is to write current graph
-Add c = tf.constant(1)
-weights is name in TS and w in Python
-Have to provide values for all placeholders. Need to feed all placeholders.
-Sess.run returns output for all the variables you requested.
