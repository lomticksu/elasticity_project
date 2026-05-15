#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>

#define E 2.0e11
#define nu 0.3

using namespace dealii;


template <int dim>
class ElasticitySolver
{
public:
    ElasticitySolver(unsigned int refinement_level);
    void run();

private:
    void create_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void output_results() const;

    const unsigned int refinement_level;

    Triangulation<dim>  triangulation;
    const FESystem<dim> fe;
    DoFHandler<dim>     dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double>       system_rhs;
    Vector<double>       solution;

    mutable TimerOutput timer;
};


template <int dim>
void right_hand_side(const std::vector<Point<dim>>& points, std::vector<Tensor<1, dim>>& values)
{
    AssertDimension(values.size(), points.size());
    Assert(dim >= 2, ExcNotImplemented());

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
    {
        for (unsigned int d = 0; d < dim; ++d)
        {
            values[point_n][d] = 0.0;
        }
    }
}

template <int dim>
ElasticitySolver<dim>::ElasticitySolver(unsigned int refinement_level)
    : refinement_level(refinement_level)
    , fe(FE_Q<dim>(1), dim)
    , dof_handler(triangulation)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
{
}


template <int dim>
void ElasticitySolver<dim>::create_grid()
{
    TimerOutput::Scope scope(timer, "1. create_grid");

    const Point<dim> p1(0.0, 0.0, 0.0);
    const Point<dim> p2(1.0, 1.0, 1.0);

    const std::vector<unsigned int> subdivisions = { 10, 10, 10 };

    GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

    for (auto& face : triangulation.active_face_iterators())
    {
        if (face->at_boundary())
        { 
            const Point<dim> center = face->center();

            if (std::abs(center[2] - 0.0) < 1e-12)
                face->set_boundary_id(1);
            else if (std::abs(center[2] - 1.0) < 1e-12)
                face->set_boundary_id(2);
        }
    }

    triangulation.refine_global(refinement_level);

    std::cout << "  Number of elements: " << triangulation.n_active_cells() << std::endl;
}


template <int dim>
void ElasticitySolver<dim>::setup_system()
{
    TimerOutput::Scope scope(timer, "2. setup_system");

    dof_handler.distribute_dofs(fe);

    const types::global_dof_index n_dofs = dof_handler.n_dofs();
    std::cout << "  Number of degrees of freedom: " << n_dofs << std::endl;

    solution.reinit(n_dofs);
    system_rhs.reinit(n_dofs);

    constraints.clear();
    VectorTools::interpolate_boundary_values(dof_handler,
                                             types::boundary_id(1),
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints);
    constraints.close();

    DynamicSparsityPattern dsp(n_dofs, n_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);
}


template <int dim>
void ElasticitySolver<dim>::assemble_system()
{
    TimerOutput::Scope scope(timer, "3. assemble_system");

    const QGauss<dim>     quadrature_formula(fe.degree + 1);
    const QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values |
                                     update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);

    const double l = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
	const double m = E / (2.0 * (1.0 + nu));

    Functions::ConstantFunction<dim> lambda(l), mu(m);

    std::vector<Tensor<1, dim>> rhs_values(n_q_points);

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
        fe_values.reinit(cell);

        cell_matrix = 0;
        cell_rhs = 0;

        lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
        mu.value_list(fe_values.get_quadrature_points(), mu_values);
        right_hand_side(fe_values.get_quadrature_points(), rhs_values);

        for (const unsigned int i : fe_values.dof_indices())
        {
            const unsigned int component_i = fe.system_to_component_index(i).first;

            for (const unsigned int j : fe_values.dof_indices())
            {
                const unsigned int component_j = fe.system_to_component_index(j).first;

                for (const unsigned int q_point : fe_values.quadrature_point_indices())
                {
                    cell_matrix(i, j) +=
                      (
                        (fe_values.shape_grad(i, q_point)[component_i] *
                         fe_values.shape_grad(j, q_point)[component_j] *
                         lambda_values[q_point])
                        +
                        (fe_values.shape_grad(i, q_point)[component_j] *
                         fe_values.shape_grad(j, q_point)[component_i] *
                         mu_values[q_point])
                        +
                        ((component_i == component_j) ?
                          (fe_values.shape_grad(i, q_point) *
                           fe_values.shape_grad(j, q_point) *
                           mu_values[q_point]) : 0)
                      ) * fe_values.JxW(q_point);
                }
            }
        }

        for (const unsigned int i : fe_values.dof_indices())
        {
            const unsigned int component_i = fe.system_to_component_index(i).first;

            for (const unsigned int q_point : fe_values.quadrature_point_indices())
            {
                cell_rhs(i) += fe_values.shape_value(i, q_point) *
                               rhs_values[q_point][component_i] *
                               fe_values.JxW(q_point);
            }
        }

        for (unsigned int face_no = 0; face_no < cell->n_faces(); ++face_no)
        {
            const auto& face = cell->face(face_no);

            if (face->at_boundary() && face->boundary_id() == 2)
            {
                fe_face_values.reinit(cell, face_no);

                for (unsigned int q = 0; q < n_face_q_points; ++q)
                {
                    const double JxW_face = fe_face_values.JxW(q);

                    const Point<dim>& qpoint = fe_face_values.quadrature_point(q);
                    const double t3 = 1.0e9 * qpoint[0];

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const unsigned int component_i = fe.system_to_component_index(i).first;

                        if (component_i == (dim - 1))
                        {
                            cell_rhs(i) += t3 *
                                           fe_face_values.shape_value(i, q) *
                                           JxW_face;
                        }
                    }
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}


template <int dim>
void ElasticitySolver<dim>::solve()
{
    TimerOutput::Scope scope(timer, "4. solve");

    SolverControl solver_control(10000, 1e-10 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionJacobi<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    std::cout << "  Last convergence: " << solver_control.last_value() << std::endl;
}


template <int dim>
void ElasticitySolver<dim>::output_results() const
{
    TimerOutput::Scope scope(timer, "5. output_results");

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    switch (dim)
    {
    case 1:
        solution_names.emplace_back("displacement");
        break;
    case 2:
        solution_names.emplace_back("x_displacement");
        solution_names.emplace_back("y_displacement");
        break;
    case 3:
        solution_names.emplace_back("x_displacement");
        solution_names.emplace_back("y_displacement");
        solution_names.emplace_back("z_displacement");
        break;
    default:
        DEAL_II_NOT_IMPLEMENTED();
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

    data_out.add_data_vector(solution, solution_names, DataOut<dim>::type_dof_data, data_component_interpretation);

    data_out.build_patches();

    const std::string filename = "solution_r" + std::to_string(refinement_level) + ".vtu";

    std::ofstream output(filename);
    data_out.write_vtu(output);

    std::cout << "  Result is in file: " << filename << std::endl;

    std::cout << "  Maximum u_" << dim  << " = " << solution.linfty_norm() << std::endl;
}


template <int dim>
void ElasticitySolver<dim>::run()
{
    std::cout << "Refinement level: " << refinement_level << std::endl;

    create_grid();
    setup_system();
    assemble_system();
    solve();
    output_results();
}


int main()
{
    try
    {
        for (unsigned int ref = 0; ref < 4; ++ref)
        {
            ElasticitySolver<3> solver(ref);
            solver.run();
        }
    }
    catch (std::exception& exc)
    {
        std::cerr << std::endl
            << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Exception on processing: " << std::endl
            << exc.what() << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
            << std::endl
            << "----------------------------------------------------"
            << std::endl;
        std::cerr << "Unknown exception!" << std::endl
            << "Aborting!" << std::endl
            << "----------------------------------------------------"
            << std::endl;
        return 1;
    }

    return 0;
}
