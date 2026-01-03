#pragma once
#ifndef MATH_AUTOGRAD_VARIABLES_HPP
#define MATH_AUTOGRAD_VARIABLES_HPP

// Include the required eigen machinery
#include "../matrix/matrix_ops.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <stack>
#include <set>
#include <map>
#include <memory>

namespace autograd {


    typedef unsigned long dim_t;

    // A set of expression types for the evaluation of the 
    // graph derivatives, each expression type generating
    // its own derivative
    enum class ExpressionType {

        // _ Constants _
        Variable,
        Constant,
        Zero,
        Identity,

        // _ Functions _
        Inversion,
        Add,
        Sub,
        Negation,
        Dot,
        ScalarMultiply,
        VectorScalarMultiply,
        Hadamard,
        Division,

        // _ Domain specific functions _ 
        Exponentiation,
        // Summation of a vector variable excluding an index
        Summation,
        LpNorm,
        SoftMaxCrossEntropy,
        // Reserved for future (?) use
        Custom

    };

    // Forward declaration just to be sure
    class ExpressionNode;

    class ExpressionNode {

    public:

        const ExpressionType type;
        // Note the reference to its own type, allowed because its just a pointer
        std::vector<ExpressionNode*> children;
        dim_t dim;
        dim_t dim_outer;

        // Anonymous union to contain all the node data, this is a 
        // bit corny but reduces storage (which is significant when handling
        // images)
        union {
            float vector_constant;
            unsigned long expr_id;
            unsigned long lp_norm;
        } expression_data;

        ExpressionNode(ExpressionType t) : type(t) { 
            dim = dim_outer = 0;
        }

        void set_value(float value) {
            expression_data.vector_constant = value;
        }

        float get_constant_value() const {
            return expression_data.vector_constant;
        }

        void set_var_dimension(dim_t dimension) {
            dim = dimension;
        }

        void set_outer_dim(dim_t outer) {
            dim_outer = outer;
        }

        void set_p_norm(unsigned long p) {
            expression_data.lp_norm = p;
        }

        std::vector<ExpressionNode*>& get_children() {
            return children;
        }

        void set_expression_id(unsigned long expr_id) {
            expression_data.expr_id = expr_id;
        }

        ExpressionType get_type() const {
            return type;
        }

        bool is_zero() const {
            return type == ExpressionType::Zero;
        }

        bool is_identity() const {
            return type == ExpressionType::Identity;
        }

        bool is_2d() const {
            return dim_outer != 0;
        }

        dim_t dimension() const {
            return dim;
        }

        dim_t outer_dimension() const {
            return dim_outer;
        }

    };

    // Very simple global owner for all nodes (no deallocation, OK for a toy base)
    class NodeGenerator {

        using VectorVariable = ExpressionNode*;

        std::vector<std::unique_ptr<ExpressionNode>> nodes;

    public:

        NodeGenerator() : nodes() { }   

        // Explicitly disallow copies, we are stories std::unique_ptr so
        // compiler forbids us from having copy operators!
        NodeGenerator(const NodeGenerator&) = delete;
        NodeGenerator& operator=(const NodeGenerator) = delete;

        std::vector<std::unique_ptr<ExpressionNode>>& all_nodes() {
            return nodes;
        }

        ExpressionNode* create_vector_variable(dim_t dimension) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Variable)
            );
            nodes.back()->set_var_dimension(dimension);
            return nodes.back().get();
        }

        ExpressionNode* create_vector_constant(dim_t dimension, float v) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Constant)
            );
            nodes.back()->set_var_dimension(dimension);
            nodes.back()->set_value(v);
            return nodes.back().get();
        }

        ExpressionNode* create_vector_op(ExpressionType type,
            const std::vector<ExpressionNode*>& children, unsigned int dimension) {

            nodes.emplace_back(
                std::make_unique<ExpressionNode>(type)
            );
            // Handle the case where no children may be present
            if (children.size()) {
                nodes.back()->children = children;
            }
            nodes.back()->set_var_dimension(dimension);
            return nodes.back().get();
        }

        ExpressionNode* identity(unsigned int dimension, unsigned int outer_dim = 0) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Identity)
            );
            auto* ret = nodes.back().get();
            ret->set_var_dimension(dimension);
            if (outer_dim)
                ret->set_outer_dim(outer_dim);
            return nodes.back().get();
        }

        ExpressionNode* zero(unsigned int dimension, unsigned int outer_dim = 0) {
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(ExpressionType::Zero)
            );
            auto* ret = nodes.back().get();
            ret->set_var_dimension(dimension);
            if (outer_dim)
                ret->set_outer_dim(outer_dim);
            return nodes.back().get();
        }

        ExpressionNode* multiply(ExpressionNode* node, double value) {
            // This always ensure that the scalar in VectorScalarMultiply is the first children.

            auto* val = create_vector_constant(/* dimension */ 1, value);
            if (node->dimension() != 1) {
                return create_vector_op(
                    ExpressionType::VectorScalarMultiply, { val, node }, node->dimension());
            }
            else {
                return create_vector_op(ExpressionType::ScalarMultiply, { node, val }, 1);
            }
        }

        ExpressionNode* multiply(double value, ExpressionNode* node) {
            return multiply(node, value);
        }


        ExpressionNode* multiply(ExpressionNode* x, ExpressionNode* y) {
            // This always ensure that the scalar in VectorScalarMultiply is the first children.
            if (x->dimension() == 1 && y->dimension() == 1)
                return create_vector_op(ExpressionType::ScalarMultiply, { x, y }, 1);
            else if (x->dimension() == 1) {
                return create_vector_op(
                    ExpressionType::VectorScalarMultiply, { x, y }, std::max(x->dimension(), y->dimension()));
            }
            else if (y->dimension() == 1) {
                return create_vector_op(
                    ExpressionType::VectorScalarMultiply, { y, x }, std::max(x->dimension(), y->dimension()));
            }
            else
                throw std::runtime_error("Operation not supported: product by two vectors makes no sense!");
            return nullptr;
        }


        ExpressionNode* create_vector_op(ExpressionType type,
            const std::initializer_list<ExpressionNode*> children, unsigned int dimension) {
            // Overload to allow more comfortable initializer list expressions
            nodes.emplace_back(
                std::make_unique<ExpressionNode>(type)
            );
            // Handle the case where no children may be present
            if (children.size()) {
                nodes.back()->children = children;
            }
            nodes.back()->set_var_dimension(dimension);
            return nodes.back().get();
        }

        ExpressionNode* squared_norm(ExpressionNode* expr) {
            return this->dot(expr, expr);
        }

        void dealloc_all() {
            nodes.clear();
        }

        inline ExpressionNode* transpose(ExpressionNode* x) {
            return nullptr;
        }

        inline ExpressionNode* dot(ExpressionNode* x, ExpressionNode* y) {
            if (x->dimension() != y->dimension()) {
                throw std::runtime_error("Cannot dot product expressions of different size");
            }
            auto* expr = create_vector_op(ExpressionType::Dot, { x, y }, 1);
            return expr;
        }

        inline ExpressionNode* negation(ExpressionNode* x) {
            return create_vector_op(ExpressionType::Negation, { x }, x->dimension());
        }

        inline ExpressionNode* exponential(VectorVariable x) {
            return create_vector_op(ExpressionType::Exponentiation, { x }, x->dimension());
        }

        inline ExpressionNode* inverse(ExpressionNode* x) {
            return create_vector_op(ExpressionType::Inversion, { x }, x->dimension());
        }

        inline ExpressionNode* sum(ExpressionNode* a, ExpressionNode* b) {
            if (a->dimension() != b->dimension())
                throw std::runtime_error("Cannot sum elements of different size");
            return create_vector_op(ExpressionType::Add, { a, b }, a->dimension());
        }

        inline ExpressionNode* sum(ExpressionNode* a, float v) {
            // Create a virtual constant from the value v e.g. create
            // a constant vector (just 1 float is stored, not O(n))
            const auto b = create_vector_constant(a->dimension(), v);
            return create_vector_op(ExpressionType::Add, { a, b }, a->dimension());
        }

        inline ExpressionNode* sub(ExpressionNode* a, ExpressionNode* b) {
            if (a->dimension() != b->dimension())
                throw std::runtime_error("Cannot sub elements of different size");
            return create_vector_op(ExpressionType::Sub, { a, b }, a->dimension());
        }

        inline ExpressionNode* sub(ExpressionNode* a, float v) {
            // Create a virtual constant from the value v e.g. create
            // a constant vector (just 1 float is stored, not O(n))
            const auto b = create_vector_constant(a->dimension(), v);
            return create_vector_op(ExpressionType::Sub, { a, b }, a->dimension());
        }

        inline ExpressionNode* prod(ExpressionNode* a, ExpressionNode* b) {
            // Elementwise product!
            if (a->dimension() != b->dimension())
                throw std::runtime_error("Cannot perform hadamard of different size");
            return create_vector_op(ExpressionType::Hadamard, { a, b }, a->dimension());
        }

        inline ExpressionNode* lp_norm(ExpressionNode* x, unsigned int p) {
            auto* node = create_vector_op(ExpressionType::LpNorm, { x }, x->dimension());
            node->set_p_norm(p);
            return node;
        }

    };



    std::string stringify_type(ExpressionType tp) {
        switch (tp) {
        case ExpressionType::Variable: return "Var";
        case ExpressionType::Add: return "Sum";
        case ExpressionType::Sub: return "Sub";
        case ExpressionType::Dot: return "Dot";
        case ExpressionType::ScalarMultiply: return "ScalarMul";
        case ExpressionType::VectorScalarMultiply: return "VectorScalarMul";
        case ExpressionType::Constant: return "Const";
        case ExpressionType::Zero: return "Zero";
        case ExpressionType::Identity: return "Identity";
        case ExpressionType::LpNorm: return "LpNorm";
        }
        return "N/a";
    }

    void print_as_binary_tree(const std::string& prefix, ExpressionNode* node, bool isLeft) {
        if (node != nullptr)
        {
            std::cout << prefix;

            std::cout << (isLeft ? "|--" : "^--");
            std::cout << stringify_type(node->get_type()) << "[" << node << "] D:";
            if (node->is_2d())
                std::cout << "( " << node->dimension() << ", " << node->outer_dimension() << ")";
            else
                std::cout << node->dimension();
            std::cout << std::endl;

            auto& children = node->get_children();
            if (children.size() > 0)
                print_as_binary_tree(prefix + (isLeft ? "|   " : "    "), children[0], true);
            if (children.size() > 1)
                print_as_binary_tree(prefix + (isLeft ? "|   " : "    "), children[1], false);
        }
    }

    void print_as_binary_tree(ExpressionNode* node) {
        print_as_binary_tree("", node, false);
    }


    class Differentiator {

        using NodeCache = std::unordered_map<ExpressionNode*, ExpressionNode*>;
        using VectorVar = ExpressionNode*;

        // Keep a differentiation frame to avoid explicit recursion, compute the
        // derivative of a node when all of its required children are ready (either
        // cached or computed before each node)
        struct DiffFrame {
            ExpressionNode* node;
            std::vector<ExpressionNode*> dchildren;
            bool ready;
        };


    public:
        template <typename ScalarType>
        ExpressionNode* differentiate(
            ExpressionNode* root,
            VectorVar variable,
            NodeGenerator& gen
        ) {
            NodeCache cache;
            return differentiate(root, variable, gen, cache);
        }

        ExpressionNode* differentiate(
            ExpressionNode* root,
            VectorVar var,
            NodeGenerator& gen,
            NodeCache& cache
        ) {
            std::stack<DiffFrame> st;
            // No children yet, root is not ready yet
            st.push({ root, {}, false });

            while (!st.empty()) {
                auto& frame = st.top();
                ExpressionNode* node = frame.node;

                std::cout << "Nodo attuale:" << std::endl;
                print_as_binary_tree(node);
                std::cout << "\n\n" << std::endl;

                // Cached?
                if (cache.count(node)) {
                    st.pop();
                    continue;
                }

                if (no_children_derivative_required(node->get_type()))
                    frame.ready = true;

                if (!frame.ready) {
                    frame.ready = true;

                    // Push children
                    for (auto* child : node->get_children()) {
                        if (!cache.count(child)) {
                            st.push({ child, {}, false });
                        }
                    }
                }
                else {
                    // Fetch all derivatives from the mapping. 
                    frame.dchildren.clear();
                    for (auto* child : node->get_children())
                        frame.dchildren.push_back(cache[child]);
                    ExpressionNode* dnode = differentiate_node(node, frame.dchildren, var, gen);
                    cache[node] = dnode;
                    st.pop();
                }
            }

            auto* der = cache[root];
            // Perform all possible optimizations on the graph. 
            unsigned int performed_opts;
            std::cout << "UNOPTIMIZED:" << std::endl;
            print_as_binary_tree(der);
            do {
                performed_opts = 0;
                optimize(der, gen, performed_opts);
            } while (performed_opts > 0);

            return cache[root];

        }

        ExpressionNode* differentiate_node(ExpressionNode* node, const std::vector<ExpressionNode*>& dchildren,
            VectorVar var, NodeGenerator& gen) {
            std::cout << "DIFFERENZIO UN NODO DI TIPO" << stringify_type(node->get_type()) << std::endl;

            if (node->get_type() == ExpressionType::Constant) {
                if ( node->dimension() == 1 )
                    return gen.zero( var->dimension() );
                else 
                    return gen.zero( var->dimension() , /* outer dim*/ node->dimension() );
            }
            else if (node->get_type() == ExpressionType::Dot) {
                auto left_d = dchildren[0];
                auto left = node->get_children()[0];
                auto right_d = dchildren[1];
                auto right = node->get_children()[1];

                return gen.sum( gen.dot(left_d, right) , gen.dot(left, right_d) );
            }
            else if (node->get_type() == ExpressionType::Variable) {
                if (node == var)
                    return gen.identity(node->dimension());
                else return gen.zero(node->dimension());
            }
            else if (node->get_type() == ExpressionType::ScalarMultiply) {
                auto left_d = dchildren[0];
                auto left = node->get_children()[0];
                auto right_d = dchildren[1];
                auto right = node->get_children()[1];

                std::cout << "dims: " << gen.multiply(left_d, right)->dimension(); 
                std::cout << "dims: " << gen.multiply(left, right_d)->dimension() << std::endl;

                return gen.sum(
                    gen.multiply(left_d, right), 
                    gen.multiply(left, right_d)
                );
            }
            else if (node->get_type() == ExpressionType::Add) {
                auto left_d = dchildren[0];
                auto right_d = dchildren[1];
                return gen.sum(left_d, right_d);
            }
            else if (node->get_type() == ExpressionType::Sub) {
                auto left_d = dchildren[0];
                auto right_d = dchildren[1];
                return gen.sub(left_d, right_d);
            }
            return nullptr;
        }

        static constexpr bool no_children_derivative_required(ExpressionType type) {
            if (type == ExpressionType::Variable || type == ExpressionType::Constant ||
                type == ExpressionType::Identity)
                return true;
            return false;
        }

        struct OptFrame {
            ExpressionNode* node = nullptr;
            size_t next_child = 0;
        };

        ExpressionNode* rewrite(ExpressionNode* node, NodeGenerator& gen, unsigned int& optimizations) {
            if (node == nullptr)
                return node;
            
            auto& children = node->get_children();
            if (node->get_type() == ExpressionType::ScalarMultiply) {
                if (children[0]->is_zero() || children[1]->is_zero())
                    return gen.zero(children[0]->dimension());
            }
            if (node->get_type() == ExpressionType::Add) {
                if (children[0]->is_zero() && children[1]->is_zero())
                    return gen.zero(children[0]->dimension());
                if (children[0]->is_zero() && !children[1]->is_zero())
                    return children[1];
                if (!children[0]->is_zero() && children[1]->is_zero())
                    return children[0];
            }
            if (node->get_type() == ExpressionType::Sub) {
                // Sub computes children[0]-children[1]
                if (children[0]->is_zero() && children[1]->is_zero())
                    return gen.zero(children[0]->dimension());
                if (children[1]->is_zero() && !children[0]->is_zero())
                    return children[0];
                if (children[0]->is_zero() && !children[1]->is_zero())
                    return gen.negation(children[1]);
            }
            if (node->get_type() == ExpressionType::Dot) {
                if (children[0]->is_zero() || children[1]->is_zero())
                    return gen.zero(children[0]->dimension());
                if (children[1]->is_identity())
                    return children[0];
                if (children[0]->is_identity())
                    return children[1];
            }
            return node;
        }

        ExpressionNode* optimize(ExpressionNode* root, NodeGenerator& gen, unsigned int& optimizations) {

            std::stack<OptFrame> stack;
            stack.push({ root, 0 });

            ExpressionNode* last_result = nullptr;
            unsigned int performed_optimizations = 0;

            while (!stack.empty()) {
                OptFrame& f = stack.top();
                ExpressionNode* node = f.node;

                if (f.next_child < node->get_children().size()) {
                    ExpressionNode* child = node->get_children()[f.next_child];
                    f.next_child++;

                    stack.push({ child, 0 });
                    continue;
                }

                ExpressionNode* new_node = rewrite(node, gen, performed_optimizations);

                stack.pop();

                if (!stack.empty()) {
                    OptFrame& parent = stack.top();
                    parent.node->get_children()[parent.next_child - 1] = new_node;
                }

                last_result = new_node;
            }
            optimizations = performed_optimizations;

            return last_result;
        }
    };



    typedef struct {

    } custom_op_t;

    static std::map<unsigned long, custom_op_t> operation_registry;

    void register_custom_operation(long op_id, void* func, void* par_der) {
        // Register the custom operation on a static private registry
        // operation_registry.insert(op_id, par_der);
    }

    // Alias for convenience template
    template <typename ScalarType> 
    using EigVec = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

    template <typename ScalarType>
    using EvalCache = std::unordered_map<ExpressionNode*, EigVec<ScalarType>>;

    // Raw buffer from eigen vector.
    template <typename ScalarType>
    using EvalMap = std::map<ExpressionNode*, 
        std::reference_wrapper<EigVec<ScalarType>>>;

    template <typename ScalarType>
    // NOTE: we make EXTENSIVE use of compilers RVO return value optimization,
    // this vector is in practice always constructed in the returnee's memory.
    EigVec<ScalarType> eval_node(
        ExpressionNode* root,
        std::vector<ExpressionNode*>& eval_ordering,
        EvalMap<ScalarType>& eval_data,
        EvalCache<ScalarType>& cache
    ) {
        EigVec<ScalarType> result(root->dimension());
       
        for (ExpressionNode* node : eval_ordering) {
            switch (node->get_type()) {
            case ExpressionType::Variable: {
                // Simply take the memory stored in eval data and copy it into the vector.
                cache[node] = eval_data.at(node).get();
                break;
            }
            case ExpressionType::Constant: {
                EigVec<ScalarType> val(node->dimension());
                val.setConstant(node->get_constant_value());
                cache[node] = std::move(val);
                break;
            }
            case ExpressionType::Sub: {
                auto* a = node->get_children()[0];
                auto* b = node->get_children()[1];
                cache[node] = cache[a] - cache[b];
                break;
            }
            case ExpressionType::Add: {
                auto* a = node->get_children()[0];
                auto* b = node->get_children()[1];
                cache[node] = cache[a] + cache[b];
                break;
            }
            case ExpressionType::ScalarMultiply: {
                auto* a = node->get_children()[0];
                auto* b = node->get_children()[1];
                // Scalar multiplication
                EigVec<ScalarType> scalar(1);
                scalar(0) = cache[a](0) * cache[b](0);
                cache[node] = scalar;
                break;
            }
            case ExpressionType::VectorScalarMultiply: {
                // Read the note in generator.multiply, we have that the scalar element
                // is always the first children. 
                auto* scalar = node->get_children()[0];
                auto* b = node->get_children()[1];
                ScalarType scalar_value = static_cast<ScalarType>(scalar->get_constant_value());
                cache[node] = scalar_value * cache[b];
            }
            case ExpressionType::Dot: {
                auto* a = node->get_children()[0];
                auto* b = node->get_children()[1];
                EigVec<ScalarType> scalar(1);
                scalar(0) = cache[a].dot(cache[b]);
                cache[node] = scalar;
            }
            }
        }
/*
        case ExpressionType::Constant: {
            // Analyze the size of the constant and create a vector of that size.
            result.assign(dim, 
                // This cast handles every reasonable scalar int, float, ...
                static_cast<ScalarType>(node->expression_data.vector_constant));
            break;
        }

        case ExpressionType::Zero: {
            // This may seem inefficient, but reasonably the computation trees are 
            // made efficient enough that this is never required.
            result.assign(dim, static_cast<ScalarType>(0));
            break;
        }

        case ExpressionType::Add: {
            if (children.size() != 2) {
                throw std::runtime_error("Add expects 2 children");
            }
            auto a = eval_node<ScalarType>(children[0], eval_data, cache);
            auto b = eval_node<ScalarType>(children[1], eval_data, cache);
            result.resize(dim);
            for (dim_t i = 0; i < dim; ++i) {
                result[i] = a[i] + b[i];
            }
            break;
        }
        // For now, we can skip Identity, Selector, Summation, etc. or throw:
        default:
            throw std::runtime_error("Evaluation not implemented for this ExpressionType");
        }

        // Move the resulting data into the cache avoiding the overhead of 
        // copy. Note that when reading a vector FROM the cache a copy is necessary
        // because we are running a DAG, e.g. multiple branches may need that data!
        */
        result = cache[root];
        std::cout << "THE RESULT IS" << result << std::endl;
        return result;
    }

    template <typename ScalarType = float>
    class VectorFunction {
        NodeGenerator node_generator;

        using VectorExpression = ExpressionNode*;
        VectorExpression root;

    public:

        void operator= (VectorExpression expr) {
            root = expr;
        }

        ExpressionNode* expr() {
            return root;
        }

        // Return a reference to its owned generator. 
        NodeGenerator& generator() {
            return node_generator;
        }

    };

    bool no_children_evaluation_required(ExpressionType type) {
        if (type == ExpressionType::Variable)
            return true;
        return false;
    }

    struct EvalFrame {
        ExpressionNode* node;
        bool ready;
    };

    void evaluate_graph_non_recursive_ordering(
        ExpressionNode* root,
        std::vector<ExpressionNode*>& node_ordering
    ) {
        std::stack<EvalFrame> st;
        // No children yet, root is not ready yet
        st.push({ root, false });

        std::set<ExpressionNode*> cache;

        while (!st.empty()) {
            auto& frame = st.top();
            ExpressionNode* node = frame.node;

            if (cache.count(node)) {
                st.pop();
                continue;
            }

            if (no_children_evaluation_required(node->get_type()))
                frame.ready = true;

            if (!frame.ready) {
                frame.ready = true;

                for (auto* child : node->get_children()) {
                    if (!cache.count(child)) {
                        st.push({ child, false });
                    }
                }
            }
            else {
                std::cout << "Pronto!" << std::endl;
                // Pseudo evaluation of the node.
                node_ordering.push_back(node);
                cache.insert(node);
                st.pop();
            }
        }
        return;
    }


    template <typename ScalarType = float>
    class ScalarFunction {

        using VectorVar = ExpressionNode*;
        using ScalarExpression = ExpressionNode*;
        using VectorExpression = ExpressionNode*;

        // A node generator object is assigned to each function for thread
        // safety and to avoid memory leaks: when the function goes out of
        // scope, all its expression nodes are destroyed. Moreover this can
        // be used to share expression nodes among different expressions for
        // different function entries (
        NodeGenerator node_generator;
        ScalarExpression root;
        std::vector<ExpressionNode*> eval_ordering;

    public:

        ScalarFunction() : root(nullptr) { }

        double operator() (EvalMap<ScalarType>& eval_data) {
            // See the notes  inside apply
            return apply(eval_data);
        }

        double apply(EvalMap<ScalarType>& eval_data) {
            if (root == nullptr) { 
                throw std::runtime_error("ScalarFunction has no root expression"); 
            } 
            if (root->dimension() != 1) { 
                throw std::runtime_error("ScalarFunction root must have dimension 1"); 
            } 
            // Note: WE CANNOT ASSUME no-aliasing of the data vectors because the expression tree is 
            // really an expression DAG because of memoization!
            // We cache intermediate portions of the graphs to avoid repeated computation of 
            // certain values.
            EvalCache<ScalarType> cache; 
            
            auto vec_result = eval_node<ScalarType>(root, eval_ordering, eval_data, cache); 
            if (vec_result.rows() != 1) { 
                throw std::runtime_error("Root evaluation did not produce a scalar, this is abnormal for a scalar function"); 
            }
            return vec_result(0);
        }

        void flush_alloc_data() {
            node_generator.dealloc_all();
        }

        // Return a reference to its owned generator. 
        NodeGenerator& generator() {
            return node_generator;
        }

        ScalarFunction& operator= (ScalarExpression expr) {
            if (expr != nullptr && expr->dimension() != 1)
                throw std::runtime_error("Cannot generate a scalar function from a vector expression!");
            root = expr;
            // This compiles the expression into a well defined evaluation ordering.
            evaluate_graph_non_recursive_ordering(expr, this->eval_ordering);
            for (int i = 0; i < eval_ordering.size(); ++i)
                std::cout << eval_ordering[i] << " ";
            std::cout << std::endl;
            // Compute now the topological ordering required for non-recursive
            // evaluation. 
            return *this;
        }

        ExpressionNode* expr() {
            return root;
        }

        void derivative(VectorFunction<ScalarType>& func, VectorVar var, unsigned int optimize_steps = 10) {
            // Compute the derivative of this scalar function with respect to a variable
            Differentiator diff;
            VectorExpression expr = diff.differentiate<ScalarType>( root, var, func.generator() );
            
            func = expr;
        }

    };


    template <typename DataType>
    std::ostream& operator<< (std::ostream& os, ScalarFunction< DataType>& sf) {
        auto* root = sf.expr();
        print_as_binary_tree(root);
        return os;
    }

    template <typename DataType>
    std::ostream& operator<< (std::ostream& os, VectorFunction< DataType>& sf) {
        auto* root = sf.expr();
        print_as_binary_tree(root);
        return os;
    }



} // ! namespace autograd

#endif //!MATH_AUTOGRAD_VARIABLES_HPP