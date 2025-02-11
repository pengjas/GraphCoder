module JC_register(
    input                clk,
    input                rst_n,
    output reg [63:0]   Q
);
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            Q <= 64'b0;
        end else begin
            Q <= Q + 1;
        end
    end
endmodule

module JC_logic(
    input                clk,
    input                rst_n,
    input [63:0]         Q,
    output reg [63:0]     Q_next
);
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            Q_next <= 64'b0;
        end else begin
            Q_next <= {Q[62:0], Q[63]};
        end
    end
endmodule

module JC_counter(
    input                clk,
    input                rst_n,
    output [63:0]       Q
);
    wire [63:0] Q_next;
    reg [63:0] Q_reg;
    JC_register u_JC_register(
        .clk(clk),
        .rst_n(rst_n),
        .Q(Q_reg)
    );
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            Q_reg <= 64'b0;
        end else begin
            Q_reg <= Q_next;
        end
    end

    wire [63:0] Q_reg_next;
    reg [63:0] Q_reg_next_reg;
    JC_logic uJC_logic(
        .clk(clk),
        .rst_n(rst_n),
        .Q(Q_reg),
        .Q_next(Q_reg_next_reg)
    );
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            Q_reg_next_reg <= 64'b0;
        end else begin
            Q_reg_next_reg <= Q_reg_next;
        end
    end

    assign Q_next = Q_reg_next[63:0];
    assign Q = Q_reg;
endmodule

module JC_counter_64(
    input                clk,
    input                rst_n,
    output [63:0]       Q
);
    wire [63:0] Q_next;
    reg [63:0] Q_reg;
    JC_register uJC_register(
        .clk(clk),
        .rst_n(rst_n),
        .Q(Q_reg)
    );

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            Q_reg <= 64'b0;
        end else begin
            Q_reg <= Q_next;
        end
    end

    assign Q_next = Q_reg + 1;
    assign Q = Q_reg;
endmodule

module JC_counter_64_2(
    input                clk,
    input                rst_n,
    output [63:0]       Q
);
    wire [63:0] Q_next;
    reg [63:0] Q_reg;
    JC_register uJC_register(
        .clk(clk),
        .rst_n(rst_n),
        .Q(Q_reg)
    );

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            Q_reg <= 64'b0;
        end else begin
            Q_reg <= Q_next;
        end
    end

    assign Q_next = Q_reg + 1;
    assign Q = Q_reg;
endmodule

module JC_counter_64_3(
    input                clk,
    input                rst
endmodule