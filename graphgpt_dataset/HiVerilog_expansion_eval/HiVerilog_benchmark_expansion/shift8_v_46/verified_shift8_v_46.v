module adder_selector (
    input [7:0] a,
    input [7:0] b,
    input ctrl,
    output reg [7:0] sum
);
    wire [7:0] adder_out;

    // Instantiating the 8-bit adder
    my_adder8 adder (.a(a), .b(b), .sum(adder_out));

    // Conditional output logic based on ctrl signal
    always @(*) begin
        if (ctrl)
            sum = adder_out;
        else
            sum = 8'b0;
    end
endmodule

module my_adder8 (
    input [7:0] a,
    input [7:0] b,
    output [7:0] sum
);
    assign sum = a + b; // Simple addition
endmodule
