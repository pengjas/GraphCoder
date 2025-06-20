`define ADD 2'b00
`define SUB 2'b01
`define AND 2'b10
`define OR  2'b11

module simple_alu(
    input clk,
    input rst,
    input [1:0] op_code,
    input [3:0] data_a,
    input [3:0] data_b,
    output [3:0] result,
    output zero
);
wire [3:0] operation_result;

op_decoder decoder (
    .op_code(op_code),
    .data_a(data_a),
    .data_b(data_b),
    .result(operation_result)
);

result_manager manager (
    .clk(clk),
    .rst(rst),
    .operation_result(operation_result),
    .final_result(result),
    .zero_flag(zero)
);

endmodule

module op_decoder(
    input [1:0] op_code,
    input [3:0] data_a,
    input [3:0] data_b,
    output reg [3:0] result
);
    always @* begin
        case(op_code)
            `ADD: result = data_a + data_b;
            `SUB: result = data_a - data_b;
            `AND: result = data_a & data_b;
            `OR: result = data_a | data_b;
            default: result = 4'b0000;
        endcase
    end
endmodule

module result_manager(
    input clk,
    input rst,
    input [3:0] operation_result,
    output reg [3:0] final_result,
    output reg zero_flag
);
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            final_result <= 0;
            zero_flag <= 1'b1;
        end else begin
            final_result <= operation_result;
            zero_flag <= (operation_result == 0);
        end
    end
endmodule