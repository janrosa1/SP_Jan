A_x1 = 1:1:3
A_x2 = 1:1:4
f(x1, x2) = log(x1+x2)
A = [f(x1,x2) for x1 in A_x1, x2 in A_x2]

B= zeros(size(A_x1)[1], size(A_x2)[1])
println(size(B))
for i in 1:size(A_x1)[1]
    for j in 1:size(A_x2)[1]

    B[i,j] = f(A_x1[i], A_x2[j])
    println(B[i,j])
    end
end

println(A.-B)

