module data_mux (
    input clk,
    input [1:0] sel,
    input [7:0] in0,
    input [7:0] in1,
    input [7:0] in2,
    input [7:0] in3,
    output [7:0] out
);

    wire [7:0] out0, out1, out2, out3;

    data_holder hold0 (
        .clk(clk),
        .data(in0),
        .out(out0),
        .sel_code(2'b00),
        .sel(sel)
    );

    data_holder hold1 (
        .clk(clk),
        .data(in1),
        .out(out1),
        .sel_code(2'b01),
        .sel(sel)
    );

    data_holder hold2 (
        .clk(clk),
        .data(in2),
        .out(out2),
        .sel_code(2'b10),
        .sel(sel)
    );

    data_holder hold3 (
        .clk(clk),
        .data(in3),
        .out(out3),
        .sel_code(2'b11),
        .sel(sel)
    );

    // Output assignment based on selection
    assign out = (sel == 2'b00) ? out0 :
                 (sel == 2'b01) ? out1 :
                 (sel == 2'b10) ? out2 :
                 out3;

endmodule

// Submodule for holding data
module data_holder (
    input clk,
    input [7:0] data,
    output reg [7:0] out,
    input [1:0] sel_code,
    input [1:0] sel
);

    always @(posedge clk) begin
        if (sel == sel_code) begin
            out <= data; // Load data if selected
        end else begin
            out <= out; // Maintain current data if not selected
        end
    end

endmodule