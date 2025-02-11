module pe(
    input clk,
    input rst,
    input [31:0] a,
    input [31:0] b,

    output [31:0] c
);
wire [63:0] mul_result;
    wire [31:0] current_value;

    
    multiplier u_multiplier(
        .clk(clk),
        .rst(rst),
        .a(a),
        .b(b),
        .mul_result(mul_result)
    );

    
    accumulator u_accumulator(
        .clk(clk),
        .rst(rst),
        .current_value(current_value),
        .mul_result(mul_result)
    );

    
    assign c = current_value;

endmodule


module multiplier(
    input clk,
    input rst,
    input [31:0] a,
    input [31:0] b,
    output [63:0] mul_result
);

    reg [63:0] mul_result_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            mul_result_reg <= 0;
        end else begin
            mul_result_reg <= a * b;
        end
    end

    assign mul_result = mul_result_reg;

endmodule


module accumulator(
    input clk,
    input rst,
    input [31:0] current_value,
    input [63:0] mul_result
);

    reg [31:0] current_value_reg;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            current_value_reg <= 0;
        end else begin
            current_value_reg <= current_value + mul_result[31:0];
        end
    end

    assign current_value = current_value_reg;

endmodule