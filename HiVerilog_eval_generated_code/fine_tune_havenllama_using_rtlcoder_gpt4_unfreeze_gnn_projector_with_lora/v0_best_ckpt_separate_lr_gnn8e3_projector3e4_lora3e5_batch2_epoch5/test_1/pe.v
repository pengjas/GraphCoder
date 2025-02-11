module pe(
    input clk,
    input rst,
    input [31:0] a,
    input [31:0] b,

    output [31:0] c
);
wire [63:0] mult_result;
    wire [31:0] current_value;

    
    multiplier u_multiplier (
        .clk(clk),
        .rst(rst),
        .a(a),
        .b(b),
        .mult_result(mult_result)
    );

    
    accumulator u_accumulator (
        .clk(clk),
        .rst(rst),
        .current_value(current_value),
        .mult_result(mult_result)
    );

    
    assign c = current_value;

endmodule


module multiplier(
    input clk,
    input rst,
    input [31:0] a,
    input [31:0] b,
    output [63:0] mult_result
);

    reg [63:0] mult_result_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mult_result_reg <= 0;
        end else begin
            mult_result_reg <= a * b;
        end
    end

    assign mult_result = mult_result_reg;

endmodule


module accumulator(
    input clk,
    input rst,
    input [31:0] current_value,
    input [63:0] mult_result,
    output [31:0] c
);

    reg [31:0] c_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            c_reg <= 0;
        end else begin
            c_reg <= c_reg + mult_result[31:0];
        end
    end

    assign c = c_reg;

endmodule