module binary_up_counter (
    input wire clk,
    input wire reset,
    output wire [3:0] out
);

    wire [3:0] current_state;

    // Submodule for count initialization
    count_init init (
        .reset(reset),
        .current_state(current_state)
    );

    // Submodule for count increment
    count_increment increment (
        .clk(clk),
        .reset(reset),
        .current_state_in(current_state),
        .current_state_out(out)
    );

endmodule

// Submodule for count initialization
module count_init (
    input wire reset,
    output reg [3:0] current_state
);

    always @ (posedge reset) begin
        current_state <= 4'b0000;
    end

endmodule

// Submodule for count increment
module count_increment (
    input wire clk,
    input wire reset,
    input wire [3:0] current_state_in,
    output reg [3:0] current_state_out
);

    always @ (posedge clk) begin
        if (reset)
            current_state_out <= 4'b0000;
        else
            current_state_out <= current_state_in + 1'b1;
    end

endmodule
