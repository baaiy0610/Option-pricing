### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ d5d0b5c0-9d58-11eb-2698-13f65e5b3f38
begin
	using Pkg
	Pkg.add("Plots")
	Pkg.add("PlutoUI")
	using PyPlot
	using PrettyTables
	using PlutoUI
	function pyplot(f;width=5,height=5)
		PyPlot.clf()
		PyPlot.grid()
		PyPlot.gcf().set_size_inches(width,height)
		PyPlot.gcf()
		PyPlot.plot3d()
		PyPlot.surface()
		
	end	
end

# ╔═╡ e62f74ce-9128-11eb-2602-c774dd606e78
md"""## Finite element method"""

# ╔═╡ 28a93d62-9129-11eb-3a4a-8349be812104
md""" **FEM**: Consider the heat equation with Dirichlet boundary conditions

$$u(\tau, x_{min}) = \bar{\alpha}(\tau), \quad u(\tau, x_{max}) = \bar{\beta}(\tau).$$
Substituting the approximation ansatz

$$u = \sum_{i=1}^{m-1}w_i(\tau)\varphi_i(x) + u_b(\tau,x), \quad u_b(\tau,x) := (\bar{\beta}(\tau) - \bar{\alpha}(\tau))\frac{x-x_{min}}{x_{max}-x_{min}} + \bar{\alpha}(\tau)$$

into the heat equation leads to the Galerkin approach.
Therefore we obtain the following system of ODEs

$$M\dot{w} + b = Bw$$
with the known matrices $M, B$ and

$$b(\tau) = \left(\int\dot{u_b}\varphi_1\,dx, \cdots, \int\dot{u_b}\varphi_{m-1}\,dx \right)^{\top}.$$

Consider a European Call-option and use its boundary conditions $\bar{\alpha}(\tau), \bar{\beta}(\tau)$. Find the approximation of the price $V$ following the steps:
*   derive the vector $b(\tau)$ analytically
*   solve the ODEs for the unknown vector $w_1, \cdots , w_{m−1}$. For this discretize time $\tau$ and $\Delta \tau = k$ from Crank-Nicolson method as:

$$(M+\frac{k}{2}B)w^{(\nu +1)} = (M-\frac{k}{2}B)w^{(\nu)} - \frac{k}{2}(b^{(\nu)}+b^{(\nu+1)})$$
* compute $w$ from the ansatz as the approximated solution of the heat equation and transform it back to the approximation of the option price

Choose some parameter values and test your code by comparing the approximated price to the exact fair price obtained from the BS formula."""

# ╔═╡ 86af8f92-9129-11eb-1f72-fddd09e4c39e
md"""## 1)Define functions that return the function to compensate the boundary conditions, namely $\varphi_b(\tau, x)$

Function to compensate for the BC FEM

- tau   : time value
- x_max : maximim space value
- x_min : minimum space value
- x     : space points
- q     : constant, q = (2r)/sigma^2
"""

# ╔═╡ ca7a1ac6-9129-11eb-21c8-b9d02d9ed6f4
function u_b(tau,x_min,x_max,x,q)
	alpha_tau=0
	beta_tau=exp.(0.5*(q+1)*x_max.+0.25*(q+1)^2*tau).-exp.(0.5*(q-1)*x_max.+0.25*(q-1)^2*tau)
	U_b=beta_tau*(x.-x_min)/(x_max-x_min)
	return U_b
end

# ╔═╡ fcc25664-912a-11eb-07ec-db72a92fb0ca
md"""## Initial condition

Function that return IC at tau = 0 for a call option, for the coefficients of the FEM method

- tau_ $0$   : time value t_$0$
- x_max : maximim space value
- x_min : minimum space value
- x     : space points
- q     : constant, q = (2r)/sigma^2
"""

# ╔═╡ 55f4c49c-912b-11eb-24d7-1f13678569a7
function c0(tau_0,x_min,x_max,x,q)
	IC = max.(exp.((q+1)*x/2).-exp.((q-1)*x/2),0)-u_b(tau_0,x_min,x_max,x,q)
	return IC
end

# ╔═╡ 534a6ae6-912e-11eb-0042-9f544f266a83
md""" ## 2)Define a function that return the vector $b(\tau, x)$ at a time point

Function that return the vector b FEM

- tau     : time value
- x_max   : maximim space value
- x_min   : minimum space value
- delta_x : space step size
- x       : space points as vector
- q       : constant, q = (2r)/sigma^2
"""

# ╔═╡ 53ef5ea4-912e-11eb-0d7c-c392aee1414b
function b(tau, x_min, x_max, delta_x,x,k)
	Beta_tau = exp.(0.5*(k+1)*x_max+0.25*(k+1)^2*tau)
	dBeta_tau = 0.25*(k+1)^2*Beta_tau
	b = ((x .- x_min)/(x_max - x_min))*delta_x*dBeta_tau
	return b
end

# ╔═╡ e49eae9e-912e-11eb-171c-7ddc32465ce7
md"""## 3)Define a function to perform the FEM using Galerkin method and the Crank-Nicolson for the ODE system, and use ansatz to get the solution of the heat equation

Function that return the solution of the heat equation using fem method

- M         : space length (without BC)
- N         : time length (without IC)
- x_min     : minimum space value
- x_max     : maximim space value
- delta_x   : space step size
- delta_tau : time step size
- tau       : time points as vector with initial point 0
- x         : space points as vector with boundary
- q         : constant, q = (2r)/sigma^2
"""

# ╔═╡ e21427a6-9130-11eb-30e3-63bd7d43d530
function B_matrix!(M,r)
	B_ma=zeros(M,M)
	for i=1:M
		B_ma[i,i]=2/r
	end
	for i=2:M
		B_ma[i,i-1]=-1/r
	end
	for i=1:M-1
		B_ma[i,i+1]=-1/r
	end
	#A[1,1]=1/r
	#A[M,M]=1/r
	return B_ma
end

# ╔═╡ e44fb474-9130-11eb-379a-198c4c817520
function M_matrix!(M,r)
	M_ma=zeros(M,M)
	for i=1:M
		M_ma[i,i]=4*r/6
	end
	for i=2:M
		M_ma[i,i-1]=r/6
	end
	for i=1:M-1
		M_ma[i,i+1]=r/6
	end
	M_ma[1,1]=2*r/6
	M_ma[M,M]=2*r/6
	return M_ma
end

# ╔═╡ a1859a11-ae54-4fad-a069-022bee6ba177
function transfer(u, M, N, K, x, tau, q)
	v =zeros(N+1, M+2)
	for i=1:N+1
		for j=1:M+2
			v[i,j] = K*u[i,j]*exp((-0.5*(q-1))*x[j]+(-0.25*(q+1)^2)*tau[i])
		end
	end
	return v
end

# ╔═╡ 2a9ad9d6-912f-11eb-14f3-3fb9f4e8509b
function G_CN_fem(M, N, x_min, x_max,delta_x,delta_tau,tau,x,q,K = 10)
	c=zeros(N+1,M)
	M_ma=M_matrix!(M,delta_x)
	B_ma=B_matrix!(M,delta_x)
	
	c[N+1,:]=c0(0,x_min,x_max,x[2:M+1],q)
	left_factor=M_ma + (delta_tau/2)*B_ma
	
	inv_left=inv(left_factor)
	
	right_factor= M_ma - delta_tau*B_ma/2
	for i=N:-1:1
		b_v = b(tau[i],x_min,x_max,delta_x,x[2:M+1],q)
		b_v1 = b(tau[i+1],x_min,x_max,delta_x,x[2:M+1],q)
		c[i,:]=inv_left*(right_factor*c[i+1,:]-delta_tau*(b_v+b_v1)/2)
	end
	w=zeros(N+1,M+2)
	w[:,1]=zeros(N+1,1)
	w[:,M+2]=u_b(tau,x_min,x_max,x[M+2],q)
	for i=1:N+1
		for k=1:M
			w[i,k+1]=c[i,k].+u_b(tau[i],x_min,x_max,x[k+1],q)			
		end
	end
	v = transfer(w, M, N, K, x, tau, q)
	return v
end

# ╔═╡ 0b38f2bc-9134-11eb-0ef7-eb8e168a60f4
begin
		sigma = 0.3
		K = 10
		S_min = 3
		S_max = 30
		r = 0.05
		T = 1
		N = 10;
		TAU = 0.5*sigma^2*T
		tau = range(TAU,0,length=N+1)
		delta_tau = TAU / N
		x_min = log(S_min/K); x_max = log(S_max/K)
		M = 511
		x = range(x_min,x_max,length=M+2)
		delta_x = (x_max-x_min)/(M+1)
		l = delta_tau / delta_x^2
		q = (2*r) / sigma^2
		u = G_CN_fem(M, N, x_min, x_max,delta_x,delta_tau,tau,x,q,K)
		#u = transfer(u1, M, N, K, x, tau, q)
		#c(tau[1],x_min,x_max,delta_x,x[2:M+1],q)
end

# ╔═╡ 1fa360c5-0904-472d-83dc-9b5054f1ffc6
begin
	include("exact.jl")
	exact = [0.,0.,0.,0.]
	for i=1:4
		exact[i] = BS_exat(sigma, r, K, 12, T)
	end
	exact
end

# ╔═╡ f0c3ed8e-1753-11ed-02da-1df2ff2ce5c9
let
	include("exact.jl");
	M_test = [500, 1000, 2000, 5000]
	N_test = [10, 10, 10, 10]
	exact = BS_exat(sigma, r, K, 12, T)
	x_min = log(S_min/K)
	x_max = log(S_max/K)
	TAU = 0.5*sigma^2*T
	data = zeros(4,5)
	
	for i= 1:4
		delta_x = (x_max - x_min) / (M_test[i] + 2)
		delta_tau = TAU / N_test[i]
		x = range(x_min,x_max,length=M_test[i]+2)
		tau = range(TAU,0,length=N_test[i]+1)
		index = Int(round((log(12/K)-log(S_min/K)) / delta_x))
		v_test = G_CN_fem(M_test[i], N_test[i], x_min, x_max,delta_x,delta_tau,tau,x,q,K)[1,index]
		data[5-i,:] = [M_test[i] N_test[i] exact v_test abs(exact-v_test)]

   	end
	
	with_terminal() do
		pretty_table(data; header = ["M", "N", "Exact", "FEM", "Error"])
	end
	
end

# ╔═╡ 2ef78440-9d80-11eb-2716-7d89abcf5637
md"""
α: $(@bind α PlutoUI.Slider(0:1:90,default=30))
β: $(@bind β PlutoUI.Slider(0:1:180,default=100))
"""

# ╔═╡ 2fc97a40-9d80-11eb-04fb-33c19e82d60c
begin
	clf()
	S = K*exp.(x)
	t = 1 .-(2/sigma^2)*tau
	v =zeros(N+1, M+2)
	suptitle("FEM")
	for i=1:N+1
		for j=1:M+2
			v[i,j] = K*u[i,j]*exp((-0.5*(q-1))*x[j]+(-0.25*(q+1)^2)*tau[i])
		end
	end
	surf(S,t,v)
	ax=gca(projection="3d")
    ax.view_init(α,β)
	
	xlabel("S")
	ylabel("t")
	zlabel("v")
	gcf()

end

# ╔═╡ 99de9c60-9dfa-11eb-1400-8b959fae8050


# ╔═╡ Cell order:
# ╟─e62f74ce-9128-11eb-2602-c774dd606e78
# ╟─28a93d62-9129-11eb-3a4a-8349be812104
# ╟─86af8f92-9129-11eb-1f72-fddd09e4c39e
# ╠═ca7a1ac6-9129-11eb-21c8-b9d02d9ed6f4
# ╟─fcc25664-912a-11eb-07ec-db72a92fb0ca
# ╠═55f4c49c-912b-11eb-24d7-1f13678569a7
# ╟─534a6ae6-912e-11eb-0042-9f544f266a83
# ╠═53ef5ea4-912e-11eb-0d7c-c392aee1414b
# ╟─e49eae9e-912e-11eb-171c-7ddc32465ce7
# ╠═e21427a6-9130-11eb-30e3-63bd7d43d530
# ╠═e44fb474-9130-11eb-379a-198c4c817520
# ╠═a1859a11-ae54-4fad-a069-022bee6ba177
# ╠═2a9ad9d6-912f-11eb-14f3-3fb9f4e8509b
# ╠═0b38f2bc-9134-11eb-0ef7-eb8e168a60f4
# ╟─d5d0b5c0-9d58-11eb-2698-13f65e5b3f38
# ╟─2ef78440-9d80-11eb-2716-7d89abcf5637
# ╟─2fc97a40-9d80-11eb-04fb-33c19e82d60c
# ╠═1fa360c5-0904-472d-83dc-9b5054f1ffc6
# ╟─f0c3ed8e-1753-11ed-02da-1df2ff2ce5c9
# ╠═99de9c60-9dfa-11eb-1400-8b959fae8050
