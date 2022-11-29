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

# ╔═╡ 5f2379e0-9d7f-11eb-1495-cb39e67a6612
begin
	using Pkg
	Pkg.add("Plots")
	Pkg.add("PlutoUI")
	Pkg.add("PyPlot")
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

# ╔═╡ 9b3c272c-96e7-11eb-3d8d-55b1108fc6bc
using LinearAlgebra

# ╔═╡ 6b3bec43-16b9-472b-9cd1-ed98f7fcdc15
md"""# Finite Volunm method"""

# ╔═╡ c5965800-9c85-11eb-2a5f-e1d6a81f067d
md"""Let $V$ denote the value of a European call or put option and let $x$ denote the price of the underlying asset. It is known that $V$ satisfies the following Black－Scholes equation:

$$LV:=\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2x^2\frac{\partial^2 V}{\partial x^2} + rx\frac{\partial V}{\partial x} - rV = 0, \quad x\geq0,\quad t \in [0, T],$$
for $$x \in I, t \in [0, T)$$, with the boundary and final(or a payoff) conditions

$$V(0,t) = g_1(t)\space\space\space\space\space\space t \in [0,T),$$
$$V(X,t) = g_2(t)\space\space\space\space\space\space t \in [0,T),$$
$$V(x,T) = g_3(t)\space\space\space\space\space\space x \in \bar{I},$$

where $I$ = $(0,X)\in \mathbb{R}, \sigma > 0$ denotes the volatility of the asset, $T > 0 $ the expiry data, r the interest rate. We assume that these given functions $g_1, g_2$ and $g_3$ defineing the above boundary and final conditions satisfy the following compatibility conditions:

$$g_3(0) = g_1(T)\space\space\space\space and \space\space\space\space g_3(X) = g_2(T)$$

The simplest way to determine the boundary conditions for call options is to choose $V(0,t) = 0$ and $V(X,t) = V(X,T)$. We may also calculate the present value of an amount received at time $T$.

$$V(0,t)=0,$$
$$V(X,t)=X-E\exp\left(\int_{t}^{T} r(\tau)d\tau\right),$$

In order to have the homogeneous Dirichlet bondary conditons, we add $f(x)=-LV_0$ to both sides of the BS-equation and define a new variable $u = V - V_0,$ where

$$V_0(x,t)=g_1(t) + \frac{g_2(t) - g_1(t)}{X}x$$

and L is the differential operator. The resulting problem can be written in the following self-ajoint form:

$$\frac{\partial u}{\partial t}- \frac{\partial}{\partial x}(ax^2\frac{\partial u}{\partial x}+bxu) + cu= f(x,t),$$

where

$$a = \frac{1}{2} \sigma^2$$
$$b = r - \sigma^2$$
$$c = 2r - \sigma^2$$

where the boundary and final conditions now become

$$u(0,t) = 0 = u(X,t), t\in [0,T)$$
$$u(x,T) = g_3(x) - V_0(x,T), x\in \bar{I}$$


"""

# ╔═╡ 9e98cec0-9caa-11eb-22a5-21ef8d8d73fd
md"""#### 1) Define functions that return the homogeneous Dirichlet boundary conditions 

Functions to return BCs:
- t : time value
- S_max : maximum space value
- K : constant
- r : constant
- T : end time

""" 

# ╔═╡ 746e557a-9610-11eb-3902-2d10caeed46c
function beta(r,t,K,S_max,T)
	return S_max.-K*exp.(-r*(T.-t))
end

# ╔═╡ 54eb63c0-9cad-11eb-084d-2bdc40662c28
md"""#### Initial condition

Functions to return initial condition at t = 0 for a call option
- x : space points
- K : constant

""" 

# ╔═╡ bfdaed7a-9610-11eb-2cf2-2f2b2d9f56b1
function initial(x,K)
	return max.(x.-K,0)
end

# ╔═╡ 9938da40-9cac-11eb-1d16-0532242ca5d2
md"""#### 2) Define functions that return $V_0$ , which helps tp denote the new variable $u = V - V_0$

Functions to return $V_0$:
- t : Time value
- S_max : maximum space value
- K : constant
- r : constant
- T : end Time
- x : space points
""" 

# ╔═╡ 2569b7a0-96e0-11eb-1250-6581b9b72006
function V_0(r,t,K,S_max,T,x)
	V_0=(S_max-K*exp(-r*(T-t)))*x/S_max
	return V_0
end

# ╔═╡ 71c43a22-9cae-11eb-1cb5-81c000e5a68f
md"""#### New Right Hand Side 
(after the BCs transfer in to homogeneous Dirichlet BCs)

Functions to return $V_0$:
- t : Time value
- S_max : maximum space value
- K : constant
- r : constant
- T : end Time
- x : space points
""" 

# ╔═╡ ec7074d0-96e3-11eb-3d7e-9bbfefcd26e6
function f(r,T,t,K,S_max,x)
	return -r*x*K*exp(-r*(T-t))/S_max
end

# ╔═╡ ee57b4e0-9cae-11eb-308c-9fdc8ff503f0
md"""#### New initial $V$ 
(after the varaiable change into $u = V -  V_0$)

Functions to return $V_0$:
- t : Time value
- S_max : maximum space value
- K : constant
- r : constant
- T : end Time
- x : space points
""" 

# ╔═╡ 18db8e0a-96e0-11eb-090b-a1612476bca0
function v_initial(r,K,S_max,T,x)
	return initial(x,K)-(S_max-K)*x/S_max
end

# ╔═╡ 14379950-9caf-11eb-18b5-8d9d906fc912
md"""#### 3) Define a funciton to perform the FVM and use Crank_Nicolson for the ODE system, and use the ansatz to get the solution of the PDE

The matrix should be :

$$\frac{u^{k+1}_i - u^{k+1}_i}{-\Delta t_k}l_i + \theta E^{k+1}_iu^{k+1}+(1-θ)E^{k}_iu^{k}=\theta f^{k+1}_i+f^{k}_i l_i$$

Functions to return the FVM matrix:
- alpha : $\frac{b}{a}$
- h : space step size
- a : constant = $\frac{1}{2} \sigma^2$
- b : constant = $r-\sigma^2$
- c : constant = $2r-\sigma^2$
- x : space points
- m : space lenth
""" 

# ╔═╡ 436d60c0-96dc-11eb-349d-07ce98e961dc
function E!(x,a,b,c,h,m,alpha)
	E=zeros(m,m);
	E[1,1]=0.25*(a+b)+b*(x[1]+x[2])*0.5*x[1]^alpha/(x[2]^alpha-x[1]^alpha)+c*h
	E[1,2]=-b*(x[1]+x[2])*0.5*x[2]^alpha/(x[2]^alpha-x[1]^alpha)
	for i=2:m
		E[i,i]=b*(x[i]+x[i-1])*0.5*x[i]^alpha/(x[i]^alpha-x[i-1]^alpha)+b*(x[i]+x[i+1])*0.5*x[i]^alpha/(x[i+1]^alpha-x[i]^alpha)+c*h
		
		E[i,i-1]=-b*(x[i]+x[i-1])*0.5*x[i-1]^alpha/(x[i]^alpha-x[i-1]^alpha)
	end
	for i=2:m-1
		E[i,i+1]=-b*(x[i]+x[i+1])*0.5*x[i+1]^alpha/(x[i+1]^alpha-x[i]^alpha)
	end
	return E
end

# ╔═╡ 0b21da08-96e0-11eb-0612-1df8f1fec7a5
function c_n_fvm(r,a,b,c,h,m,n,alpha,s,delta_t,T,S_max,t,K)
	v=zeros(n+1,m)
	v[n+1,:]=v_initial(r,K,S_max,T,s[2:m+1])
	id=1*Matrix(I,m,m)
	E=E!(s[2:m+2],a,b,c,h,m,alpha)
	left_factor=id*h/delta_t+0.5*E
	right_factor=id*h/delta_t-0.5*E
	inv_left=inv(left_factor)
	for i=n:-1:1
		f_1=f(r,T,t[i+1],K,S_max,s[2:m+1])
		f_0=f(r,T,t[i],K,S_max,s[2:m+1])
		v[i,:]=inv_left*(right_factor*v[i+1,:]+0.5*(f_1+f_0)*h)
	end
	w=zeros(n+1,m+2)
	boundary=beta(r,t,K,S_max,T)
	for i=n+1:-1:1
		w[i,m+2]=boundary[n+2-i]
		for k=1:m
			w[i,k+1]=v[i,k]+V_0(r,t[n+2-i],K,S_max,T,s[k+1])
		end
	end
	return w	
end

# ╔═╡ db1e5e9e-9610-11eb-0359-f90b80af1aa7
begin
	r=0.05
	sigma=0.3
	a=0.5*sigma^2
	b=r-sigma^2
	c=2r-sigma^2
	alpha=b/a
	m=511
	n=10
	S_min=0
	S_max=30
	K=10
	s=range(S_min,S_max,length=m+2)
	h=(S_max-S_min)/(m+2)
	T=1
	t=range(T,0,length=n+1)
	delta_t=T/(n+1)
	v = c_n_fvm(r,a,b,c,h,m,n,alpha,s,delta_t,T,S_max,t,K)
end

# ╔═╡ 54a05c52-9610-11eb-2ae9-3fe028ccc25f
function alpha_b()
	return zeros(1,n+1)
end

# ╔═╡ 253a80d3-b1c4-4293-8df7-1b367a2c12f9
begin
	include("exact.jl")
	exact = [0.,0.,0.,0.]
	for i=1:4
		exact[i] = BS_exat(sigma, r, K, 12, T)
	end
	exact
end

# ╔═╡ ed271120-9d7f-11eb-2ac4-5509c919805e
md"""
α: $(@bind α PlutoUI.Slider(0:1:90,default=30))
β: $(@bind β PlutoUI.Slider(0:1:180,default=100))
"""

# ╔═╡ e8127500-9ca6-11eb-3273-55f23cd83d2d
let
	clf()
	surf(s,t,v)
	ax=gca(projection="3d")
    ax.view_init(α,β)
	suptitle("FVM")
	xlabel("S")
	ylabel("t")
	zlabel("v")
	gcf()
	
end

# ╔═╡ 7d5c0911-4f17-4ea6-9a11-d04b60f5b4ac
begin
	R = 206/513
	M_test = [1000,500,300,200]
	N_test = [150,100,50,10]
	
	Error = [0.,0.,0.,0.]
	v_test = [0.,0.,0.,0.]
	for i = 1:length(N_test)
		S_test=range(S_min,S_max,length=M_test[i]+2)
		h_t=(S_max-S_min)/(M_test[i]+2)
		t_2=range(T,0,length=N_test[i]+1)
		delta_t_2=T/(N_test[i]+1)
		v_test[i] = c_n_fvm(r,a,b,c,h_t,M_test[i],N_test[i],alpha,S_test,delta_t_2,T,S_max,t_2,K)[1,Int(round(R*(M_test[i]+2)))]
		Error[i] = abs(v_test[i] - exact[i])
		#Int(round(R*(M_test[i]+2)))
	end
end 

# ╔═╡ 51cde43a-88a3-4ca2-8223-dae7c9cf1016
begin
	data = hcat( M_test,N_test, exact, v_test, Error)
	header = (["M","N","Exact","FVM","Error"])
	with_terminal() do
       	pretty_table(data;header=header)
	end
end


# ╔═╡ Cell order:
# ╟─6b3bec43-16b9-472b-9cd1-ed98f7fcdc15
# ╟─c5965800-9c85-11eb-2a5f-e1d6a81f067d
# ╠═9b3c272c-96e7-11eb-3d8d-55b1108fc6bc
# ╟─9e98cec0-9caa-11eb-22a5-21ef8d8d73fd
# ╠═54a05c52-9610-11eb-2ae9-3fe028ccc25f
# ╠═746e557a-9610-11eb-3902-2d10caeed46c
# ╟─54eb63c0-9cad-11eb-084d-2bdc40662c28
# ╠═bfdaed7a-9610-11eb-2cf2-2f2b2d9f56b1
# ╟─9938da40-9cac-11eb-1d16-0532242ca5d2
# ╠═2569b7a0-96e0-11eb-1250-6581b9b72006
# ╟─71c43a22-9cae-11eb-1cb5-81c000e5a68f
# ╠═ec7074d0-96e3-11eb-3d7e-9bbfefcd26e6
# ╟─ee57b4e0-9cae-11eb-308c-9fdc8ff503f0
# ╠═18db8e0a-96e0-11eb-090b-a1612476bca0
# ╟─14379950-9caf-11eb-18b5-8d9d906fc912
# ╠═436d60c0-96dc-11eb-349d-07ce98e961dc
# ╠═0b21da08-96e0-11eb-0612-1df8f1fec7a5
# ╠═db1e5e9e-9610-11eb-0359-f90b80af1aa7
# ╟─5f2379e0-9d7f-11eb-1495-cb39e67a6612
# ╟─ed271120-9d7f-11eb-2ac4-5509c919805e
# ╟─e8127500-9ca6-11eb-3273-55f23cd83d2d
# ╠═253a80d3-b1c4-4293-8df7-1b367a2c12f9
# ╠═7d5c0911-4f17-4ea6-9a11-d04b60f5b4ac
# ╟─51cde43a-88a3-4ca2-8223-dae7c9cf1016
