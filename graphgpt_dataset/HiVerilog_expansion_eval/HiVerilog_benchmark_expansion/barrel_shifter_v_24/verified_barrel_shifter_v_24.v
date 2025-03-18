module priority_encoder(in, code, valid);
    input [7:0] in;
    output [2:0] code;
    output valid;
    wire [7:0] found;

    // Instance for each bit checking
    check_bit cb7 (.in(in[7]), .found(found[7]));
    check_bit cb6 (.in(in[6]), .found(found[6]));
    check_bit cb5 (.in(in[5]), .found(found[5]));
    check_bit cb4 (.in(in[4]), .found(found[4]));
    check_bit cb3 (.in(in[3]), .found(found[3]));
    check_bit cb2 (.in(in[2]), .found(found[2]));
    check_bit cb1 (.in(in[1]), .found(found[1]));
    check_bit cb0 (.in(in[0]), .found(found[0]));

    assign valid = |in;  // valid is high if any bit in 'in' is high
    assign code = found[7] ? 3'b111 :
                  found[6] ? 3'b110 :
                  found[5] ? 3'b101 :
                  found[4] ? 3'b100 :
                  found[3] ? 3'b011 :
                  found[2] ? 3'b010 :
                  found[1] ? 3'b001 :
                  found[0] ? 3'b000 : 3'b000;

endmodule

module check_bit(in, found);
    input in;
    output found;
    assign found = in;  // This module will output '1' if input is '1'
endmodule
