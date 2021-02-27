clear all
close all

% parameters
y = [1 3];
phi = [0.8 0.2; 0.2 0.8];
beta = 0.95;

% bond
b_1 = -15;
b_N = 60;
N = 151;
b = linspace(b_1,b_N,N)';

% tolerance
tol_B = 0.0001;
tol_V = 0.0001;

% max iteration
max_iter_B = 100;
max_iter_V = 100;

q = beta;
adjust = 0.0001;
test_B = 1;
iter_B = 1;
while test_B > tol_B && iter_B <= max_iter_B
    
    % utility function
    u = NaN(N,N,2);
    for i = 1:N
        for j = 1:N
            for k = 1:2
                if y(k) + b(i) - q*b(j) > 0
                    u(i,j,k) = log(y(k) + b(i) - q*b(j));
                end
            end
        end
    end
    
    % value function iteration
    V = zeros(N,2);
    V_new = zeros(N,2);
    b_policy = zeros(N,2);
    test_V = 1;
    iter_V = 1;
    while test_V > tol_V && iter_V <= max_iter_V
        [V_new(:,1),b_policy(:,1)] = max(u(:,:,1) + beta*phi(1,1)*ones(N,1)*V(:,1)' + beta*phi(2,1)*ones(N,1)*V(:,2)',[],2);
        [V_new(:,2),b_policy(:,2)] = max(u(:,:,2) + beta*phi(1,2)*ones(N,1)*V(:,1)' + beta*phi(2,2)*ones(N,1)*V(:,2)',[],2);
        test_V = max(max(abs(V_new - V)));
        V = V_new;
        iter_V = iter_V + 1;
    end
    
    % transition matrix
    A = zeros(2*N,2*N);
    for i = 1:N
        A(i,b_policy(i,1)) = phi(1,1);
        A(i,b_policy(i,1) + N) = phi(1,2);
        A(i + N,b_policy(i,2)) = phi(2,1);
        A(i + N,b_policy(i,2) + N) = phi(2,2);
    end
    
    % stationary distribution
    [f, eig_value] = eig(A');
    [~,x] = min(abs(diag(eig_value) - 1));
    stationary_dist = f(:,x)/sum(f(:,x));
    
    % bond market clearing
    B = b'*(stationary_dist(1:N) + stationary_dist(N+1:2*N));
    
    %adjust bond price
    q = q + adjust*B;
    
    test_B = abs(B);
    iter_B = iter_B + 1;
end